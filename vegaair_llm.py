import openai
import json
import numpy as np
from typing import List
from pathlib import Path
from utils.text_ocr import txt_loss
from utils.visual_density import vd_loss, overlap_loss
from utils.metrics import wave_metric
from openai import OpenAI
import logging
import time
import os
from utils.salformer_utils import init_network, predict
import argparse
from utils.utils import update_chart, load_json, init_bo_params
import env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/vegaair_llm.log"),
        logging.StreamHandler()
    ]
)

def load_vega_lite_schema(file_path):
    """
    Load a Vega-Lite JSON schema from the given file path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            schema = json.load(file)
        logging.info(f"Loaded schema from {file_path}")
        return schema
    except Exception as e:
        logging.error(f"Failed to load schema from {file_path}: {e}")
        return None
    
def parse_gpt_response(response_text):
    """
    Parse the GPT response text into a Python dictionary.
    """
    try:
        # Attempt to extract JSON from the response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        json_str = response_text[start:end]
        analysis = json.loads(json_str)
        logging.info("Parsed GPT response successfully")
        return analysis
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to parse GPT response as JSON: {e}")
        logging.debug(f"GPT Response: {response_text}")
        return None

def save_analysis(output_path, analysis):
    """
    Save the analysis dictionary as a JSON file to the specified output path.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(analysis, file, indent=2)
        logging.info(f"Saved analysis to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save analysis to {output_path}: {e}")


def generate_prompt(schema_json, params: dict = {},  generate_full: bool = False, loss: float = 0.):
    schema = {
        "x0": float,
        "x1": float,
        "x2": float,
        "x3": float,
        "x4": float,
        "x5": float,
        "x6": float,
        "x7": float,
        "x8": float,
        "x9": float,
        "x_rt": int,
        "x_ot": int
    }
    if generate_full:
        prompt = f"""You are an expert in data visualization. Given the following Vega-Lite chart specification, suggest improvements to enhance readability, color harmony, and data accuracy. 
    Current Vega-Lite Specification:
    {json.dumps(schema_json, indent=2)}

    Optimize the paramters such that negative loss is maximized and Provide the modified parameter dictionary.
    Current Loss: {loss}
    Only change the following parameters:
    {params}
    Please respond only with the new parameters dictionary in JSON format. Follow this schema exactly and use double quotes for property names:
    {schema}
    """
    else:
         prompt = f"""You are an expert in data visualization. Given the following Vega-Lite chart specification, suggest improvements to enhance readability, color harmony, and data accuracy. 
    Current Vega-Lite Specification:
    {json.dumps(schema_json, indent=2)}

    Optimize the paramters such that negative loss is maximized and Provide the modified parameter dictionary.
    Current Loss: {loss}
    Only change the following parameters:
    {params}
    Please respond only with the new parameters dictionary in JSON format. Follow this schema exactly and use double quotes for property names:
    {schema}
    """
    return prompt


def process_file(input_file, output_folder):
    """
    Process a single Vega-Lite schema file and save the GPT analysis.
    """
    schema = load_vega_lite_schema(input_file)
    if not schema:
        logging.error(f"Skipping file due to load failure: {input_file}")
        return

    prompt = generate_prompt(schema)
    gpt_response = llm_optim_func(prompt)
    if not gpt_response:
        logging.error(f"Skipping file due to GPT API failure: {input_file}")
        return

    analysis = parse_gpt_response(gpt_response)
    if not analysis:
        logging.error(f"Skipping file due to parsing failure: {input_file}")
        return

    # Determine the output file path
    input_filename = Path(input_file).stem
    output_file = Path(output_folder) / f"{input_filename}_analysis.json"
    save_analysis(output_file, analysis)

def optim_func(predictions: List, bboxes: List[np.ndarray], chart_json: json, params: dict) -> dict:
    """
    Optimisation function of BO.

    Args:
        predictions[list]: same as the output of predict()
        predictions[0]: heatmap from VisSalFormer (np.array)
        predictions[1]: original image (Image)
        predictions[2]: gray image (np.array)
        bbox: bounding box coordinates

    Returns: score of the optimisation function
    """
    from utils.text_ocr import txt_loss
    from utils.visual_density import vd_loss, overlap_loss
    from utils.metrics import wave_metric
    # Text loss is a metric that measures the readability of texts, ranges [0, 1]    
    TXT_OCR = txt_loss(predictions[1], chart_json) * 512.
    # WAVE is a metric that measures how close the colors in the image are to the preferred colors from human [0, 1]
    WAVE = wave_metric(predictions[1]) * 256.
    VD = vd_loss(predictions[2]) * 768.
    # heatmap_mean is the mean value of saliency maps in the bounding box (larger than 8, which thresholds the whitespaces out)
    heatmap_mean = 0
    for bbox in bboxes:
        bbox_heapmap = predictions[0][bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if bbox_heapmap[bbox_heapmap>8].size > 0:
            heatmap_mean += np.mean(bbox_heapmap[bbox_heapmap>8]) # thresholding the low salient pixels, so that the size of bounding box won't matter that much
    if len(bboxes) == 0:
        return {"loss_max": (WAVE + TXT_OCR - VD - 256 - 1024, 0)}
    # Overlap loss is a metric that penalise the too thick conditions: 0 for no overlap, 1 for overlap
    OVERLAP = overlap_loss(bboxes[0],
                            chart_json['vconcat'][0],
                            len(chart_json['vconcat'][0]['data']['values']),
                            params) * 1024.
    return {"loss_max": (WAVE + TXT_OCR + 8 * heatmap_mean / len(bboxes) - VD - OVERLAP, 0)}

client = None
use_ollama = False
def get_llm_client(model: str = 'llama'):
    global client
    global use_ollama
    if model == 'llama':
        client = OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='ollama')
        use_ollama = True
    else:
        client = OpenAI()
        # Initialize OpenAI API key
        openai.api_key = os.getenv('OPENAI_API_KEY')

def get_new_chart_llm(prompt: str, current_chart_json: dict, prompt_num: int = 0, conversation: list = []) -> dict:
    """
    LLM-based optimization function for Vega-Lite charts.

    Args:
        prompt (string): Current Vega-Lite chart prompt
        current_chart_json (dict): Current Vega-Lite chart specification.
        params (dict): Parameters that might influence the LLM's suggestions.

    Returns:
        dict: Suggested improved Vega-Lite chart specification.
    """
    retries = 3
    for attempt in range(retries):
        try:
            if prompt_num == 0:
                response = client.chat.completions.create(
                            model='llama3.2:1b' if use_ollama else 'gpt-4o',
                            messages=conversation,
                            temperature=0.3,
                            max_tokens=2048,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                        )
            else:
                response = client.chat.completions.create(
                    model='llama3.2:1b' if use_ollama else 'gpt-4o',
                    messages=conversation,
                    temperature=0.3,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            suggestion = response.choices[0].message.content.strip()
            print(suggestion)
            suggestion = suggestion.replace('\n','').replace('json','').replace('`','')
            updated_parameters = json.loads(suggestion)
            logging.info("Received response from LLM")
            logging.info(f"{updated_parameters}")

            #Save Optimization History:
            conversation.append({"role": "assistant", "content": suggestion})
            return updated_parameters, conversation
        except Exception as e:
            #import IPython; IPython.embed()
            logging.error(f"OpenAI API error: {e}")
            break
    logging.error("Failed to retrieve response from GPT API after multiple attempts.")
    return None


def llm_based_optim(chart_json: dict, annotation: dict, query: dict, optim_path: str, chart_name: str, max_iter: int = 5, is_multi: bool = False):
    Path(optim_path).mkdir(parents=True, exist_ok=True)
    
    best_chart = chart_json
    best_loss = float('inf')  # Assuming higher scores are better
    loss = best_loss
    parameters = init_bo_params()
    conversation = []

    system_message ={
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": f"""You are an expert in data visualization. Given the following Vega-Lite chart specification, suggest improvements to enhance readability, color harmony, and data accuracy. Provide the modified parameter dictionary. Optimize the paramters such that negative loss is maximized.
                                    Only change the following parameters:
                                    {parameters}
                                    Respond only with the new paramters dictionary in JSON format."""
            }
            ]
        }
    conversation.append(system_message)
                            
    #Max Iterations for improving one chart sample: Default 10
    for i in range(max_iter):
        print(f"Optimization iteration {i+1}/{max_iter}")
        # Generate improved chart using LLM
        prompt = generate_prompt(chart_json, params=parameters, generate_full=True if i == 0 else False, loss=loss if type(loss) == float else loss['loss_max'][0])
        conversation.append({"role":"user", "content":prompt})
        updated_parameters, conversation = get_new_chart_llm(prompt, chart_json, i, conversation)  # Pass any relevant params
        #Evaluate improved chart:
        # PARAMS: [aspect_ratio, font_size_axis, font_size_mark, bar_size(bandwidth), hl_color_h, hl_color_s, hl_color_v, bg_color_h, bg_color_s, bg_color_v, axis_label_rotation(v_bar), orientation]
        updated_chart_json, bboxes = update_chart(chart_json, updated_parameters, annotation, query, is_multi=is_multi)
        predictions = predict(model, image_processor, tokenizer, query['question'])
        loss = optim_func(predictions, bboxes, updated_chart_json, updated_parameters)
        logging.info(f"Loss: {loss}")

        # Update best chart if improved
        if loss['loss_max'][0] < best_loss:
            best_loss = loss['loss_max'][0]
            best_chart = updated_chart_json
            # Optionally save intermediate results
            with open(Path(optim_path) / f"{chart_name}_iter_{i+1}.json", "w") as f:
                json.dump(best_chart, f, indent=2)

    # Save the best chart
    with open(Path(optim_path) / f"{chart_name}_best.json", "w") as f:
        json.dump(best_chart, f, indent=2)

    print(f"Optimization completed. Best loss: {best_loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/defaults/4488.json")
    parser.add_argument("--annot_path", type=str, default="./data/annotations/4488.json")
    parser.add_argument("--optim_path", type=str, default="./data/optims")
    parser.add_argument("--model_path", type=str, default="./models/model_lr6e-5_wd1e-4.tar")
    parser.add_argument('--llm', type=str, default="gpt-4o")
    parser.add_argument('--is_multi', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = vars(parser.parse_args())

    model, image_processor, tokenizer = init_network()
    llm = args['llm']

    get_llm_client(llm)

    if '.json' in args['data_path']:
        chart_json, annot_json = load_json(args['data_path'], args['annot_path'])
        #bayesian_optim(chart_json, annot_json, query=annot_json['tasks'][0], optim_path=args['optim_path'], chart_name=args['data_path'].split('/')[-1].strip('.json'), is_multi=args['is_multi'])
        llm_based_optim(chart_json, annot_json, query=annot_json['tasks'][0], optim_path=args['optim_path'], chart_name=args['data_path'].split('/')[-1].strip('.json'), is_multi=args['is_multi'])
    else: # batch processing
        for data_json in os.listdir(args['data_path']):
            if not data_json.endswith('.json'): continue
            if not args['overwrite'] and os.path.exists(os.path.join(args['optim_path'],data_json)): continue
            chart_json, annot_json = load_json(os.path.join(args['data_path'], data_json), os.path.join(args['annot_path'], data_json))
            #bayesian_optim(chart_json, annot_json, query=annot_json['tasks'][0], optim_path=args['optim_path'], chart_name=data_json.strip('.json'), is_multi=args['is_multi'])
            #llm_based_optim(chart_json, annot_json, query=annot_json['tasks'][0], optim_path=args['optim_path'], chart_name=args['data_path'].split('/')[-1].strip('.json'), is_multi=args['is_multi'])

