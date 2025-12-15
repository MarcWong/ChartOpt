import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
from pathlib import Path
torch.manual_seed(42)
import numpy as np
np.random.seed(42)
from typing import List
from utils.utils import update_chart, load_json, init_bo_params
from utils.salformer_utils import init_network, predict

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models
# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.surrogate import Surrogate
# Experiment examination utilities
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
# BoTorch components
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy

def optim_func(predictions: List, bboxes: List[np.ndarray], chart_json: dict, params: dict, is_multi: bool = False, is_pie: bool = False) -> dict:
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
    assert chart_json['vconcat'][0]['layer'][0]['mark']['type'] in ['arc', 'bar', 'line']
    if chart_json['vconcat'][0]['layer'][0]['mark']['type'] == 'arc':
        weights = {'txt_ocr': 512., 'wave': 0, 'vd': 768., 'overlap': 2048.}
    elif chart_json['vconcat'][0]['layer'][0]['mark']['type'] == 'bar':
        if is_multi:
            weights = {'txt_ocr': 512., 'wave': 512., 'vd': 768., 'overlap': 2048.}
        else:
            weights = {'txt_ocr': 512., 'wave': 256., 'vd': 768., 'overlap': 2048.}
    else: # line chart
        weights = {'txt_ocr': 512., 'wave': 0., 'vd': 768., 'overlap': 0}

    from utils.text_ocr import txt_loss
    from utils.visual_density import vd_loss, overlap_loss
    from utils.metrics import wave_metric
    # Text loss is a metric that measures the readability of texts, ranges [0, 1]    
    TXT_OCR = txt_loss(predictions[1], chart_json) * weights['txt_ocr']
    VD = vd_loss(predictions[2]) * weights['vd']
    # Overlap loss is a metric that penalise the overlapping conditions: 0 for no overlap, 1 for overlap    
    OVERLAP = overlap_loss(chart_json, params, is_multi, is_pie) * weights['overlap']
    # WAVE is a metric that measures how close the colors in the image are to the preferred colors from human [0, 1]
    if chart_json['vconcat'][0]['layer'][0]['mark']['type'] != 'bar':
        WAVE = 0
    else:
        WAVE = wave_metric(predictions[1]) * weights['wave']
    if len(bboxes) == 0:
        return {"loss_max": (WAVE + TXT_OCR - VD - OVERLAP, 0)}
    # heatmap_mean is the mean value of saliency maps in the bounding box (larger than 8, which thresholds the whitespaces out)
    heatmap_mean = 0
    for bb in bboxes:
        bbox = bb['bbox']
        bbox_heapmap = predictions[0][bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if bbox_heapmap[bbox_heapmap>8].size > 0:
            heatmap_mean += np.mean(bbox_heapmap[bbox_heapmap>8]) # thresholding the low salient pixels, so that the size of bounding box won't matter that much
    return {"loss_max": (WAVE + TXT_OCR + 4 * heatmap_mean / len(bboxes) - VD - OVERLAP, 0)}

def bayesian_optim(chart_json: dict, annotation: dict, query: dict, optim_path: str, chart_name:str, max_iter: int = 200, is_multi: bool = False, is_pie: bool = False):
    print('Starting Bayesian optimization for chart:', chart_name)
    Path.mkdir(Path(optim_path), exist_ok=True)
    gs = GenerationStrategy(
        steps=[
            GenerationStep(  # Initialization step
                # Which model to use for this step
                model=Models.SOBOL,
                # How many generator runs (each of which is then made a trial) to produce with this step
                num_trials=16,
                # How many trials generated from this step must be `COMPLETED` before the next one
                min_trials_observed=5,
            ),
            GenerationStep(  # BayesOpt step
                model=Models.BOTORCH_MODULAR,
                # No limit on how many generator runs will be produced
                num_trials=max_iter,
                model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                    "surrogate": Surrogate(SingleTaskGP),
                    "botorch_acqf_class": ExpectedImprovement,
                },
            ),
        ]
    )
    ax_client = AxClient(generation_strategy=gs)
    parameters = init_bo_params(is_multi, is_pie)
    ax_client.create_experiment(
        name="baropt_experiment",
        parameters=parameters,    
        objectives={"loss_max": ObjectiveProperties(minimize=False)}
    )

    # Optimization loop
    parameterization, trial_index = ax_client.get_next_trial()
    chart_json, bboxes, png_buffer = update_chart(chart_json, parameterization, annotation, query, is_multi=is_multi, is_pie=is_pie)
    ax_client.complete_trial(trial_index=trial_index, raw_data={"loss_max": (-100000, 0)})
    for i in range(max_iter):
        parameterization, trial_index = ax_client.get_next_trial()
        print('Trial {}: {}'.format(trial_index, parameterization))
        predictions = predict(model, image_processor, tokenizer, png_buffer, query['question'])
        chart_json, bboxes, png_buffer = update_chart(chart_json, parameterization, annotation, query, is_multi=is_multi, is_pie=is_pie)
        ax_client.complete_trial(trial_index=trial_index, raw_data=optim_func(predictions, bboxes, chart_json, parameterization, is_multi=is_multi, is_pie=is_pie))

    best_parameters, values = ax_client.get_best_parameters()
    update_chart(chart_json, best_parameters, annotation, query, optim_path, chart_name, is_multi=is_multi, is_pie=is_pie)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/defaults/4488.json")
    parser.add_argument("--annot_path", type=str, default="./data/annotations/4488.json")
    parser.add_argument("--optim_path", type=str, default="./data/optims")
    parser.add_argument("--model_path", type=str, default="./models/model_lr6e-5_wd1e-4.tar")
    parser.add_argument('--is_multi', action='store_true')
    parser.add_argument('--is_pie', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = vars(parser.parse_args())

    model, image_processor, tokenizer = init_network()

    if '.json' in args['data_path']:
        chart_json, annot_json = load_json(args['data_path'], args['annot_path'])
        bayesian_optim(chart_json, annot_json, query=annot_json['tasks'][0], optim_path=args['optim_path'], chart_name=args['data_path'].split('/')[-1].strip('.json'), is_multi=args['is_multi'], is_pie=args['is_pie'])
    else: # batch processing
        for data_json in os.listdir(args['data_path']):
            print('Working on data path:', data_json)
            if not data_json.endswith('.json'): continue
            if not args['overwrite'] and os.path.exists(os.path.join(args['optim_path'],data_json)): continue
            chart_json, annot_json = load_json(os.path.join(args['data_path'], data_json), os.path.join(args['annot_path'], data_json))
            bayesian_optim(chart_json, annot_json, query=annot_json['tasks'][0], optim_path=args['optim_path'], chart_name=data_json.strip('.json'), is_multi=args['is_multi'], is_pie=args['is_pie'])
