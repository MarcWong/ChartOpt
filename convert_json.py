import os, sys
import argparse
import json
import re
from typing import List
from tqdm import trange
from utils.utils import save_chart_batch
from decimal import Decimal
from pathlib import Path

def load_base_json(base_json) -> dict:
    f = open(base_json)
    base_json = json.load(f)
    return base_json.copy()

def sortby_value(data_entries: List, v_type: str, is_multi: bool) -> List:
    if v_type == 'h_bar' and not is_multi:
        sorted_data = sorted(data_entries, key=lambda x:float(x['value']), reverse=True)
        return list(sorted_data)
    # elif v_type == 'v_bar':
    #     return data_entries
    else:
        return data_entries

def write_tasks(annot_json: dict, questions: List, input_path: str, base_path: str, output_path: str, filename:str, v_type: str, is_multi: bool):
    if is_multi:
        if not len(annot_json['models']) == 2: return
        if v_type == 'h_bar':
            output_json = load_base_json(os.path.join(base_path, 'hbar_muliticol_template.json'))
        else:
            return
    else:
        if len(annot_json['models']) > 1: return
        if v_type == 'h_bar':
            output_json = load_base_json(os.path.join(base_path, 'hbar_template.json'))
        elif v_type == 'v_bar':
            output_json = load_base_json(os.path.join(base_path, 'vbar_template.json'))
        else:
            return

    if not isinstance(annot_json['models'], list) or not annot_json['models'][0]: return   

    annot_json['tasks'], data_entries = [], []
    # if 'title' in annot_json['general_figure_info'].keys():
    #     output_json['vconcat'][0]['title'] = annot_json['general_figure_info']['title']['text']
    # else:
    #     output_json['vconcat'][0]['title'] = ''
    for mm, annot_model in enumerate(annot_json['models']):
        for ii, q in enumerate(questions):
            entities, ariaLabels = [], []
            q['label'] = re.sub('[{()}]', '', q['label'])
            q_labels = q['label'].split(',')
            for q_label in q_labels:
                lowest_label = ''
                lowest_value = 100000000
                highest_label = ''
                highest_value = -1
                for i, x_l in enumerate(annot_model['x']):
                    x_label = str(x_l).replace("'", "")
                    value = re.sub('[^0-9.]','', str(annot_model['y'][i]))
                    try:
                        float(value)
                    except ValueError:
                        return
                    if not value: return
                    if value.find('.0') > -1:
                        value = int(Decimal(value))
                    elif value.find('0.') > -1 or value.find('.') > -1:
                        if len(value) - value.find('.') > 2:
                            value = float(Decimal(value).quantize(Decimal('.01'))) # round up to 2 decimal places
                    else:
                        if float(value) % 1 == 0:
                            value = int(float(value))
                        else:
                            value = float(value)
                    if ii == 0:
                        if is_multi:
                            data_entries.append({"Entity": x_label, "group": annot_model['name'], "value": value})
                        else:
                            data_entries.append({"Entity": x_label, "value": value})
                    if x_label.lower() in q_label.lower() or q_label.lower() in x_label.lower() or x_label.lower() in q['query'].lower():
                        entities.append(x_label)
                        if v_type == 'h_bar':
                            label = f"value: {value}; Entity: {x_label}"
                        elif v_type == 'v_bar':
                            label = f"Entity: {x_label}; value: {value}"
                        if is_multi:
                            label += f"; group: {annot_model['name']}"
                        ariaLabels.append(label)

                    if float(value) < float(lowest_value):
                        lowest_label = x_label
                        lowest_value = float(value)
                    if float(value) > float(highest_value):
                        highest_label = x_label
                        highest_value = float(value)
                if lowest_label and ('least' in q['query'].lower() or 'lowest' in q['query'].lower()):
                    entities.append(lowest_label)
                    if v_type == 'h_bar':
                        label = f"value: {lowest_value}; Entity: {lowest_label}"
                    elif v_type == 'v_bar':
                        label = f"Entity: {lowest_label}; value: {lowest_value}"
                    if is_multi:
                        label += f"; group: {annot_model['name']}"
                        ariaLabels.append(label)
                if highest_label and ('most' in q['query'].lower() or 'highest' in q['query'].lower()):
                    entities.append(highest_label)
                    if v_type == 'h_bar':
                        label = f"value: {highest_value}; Entity: {highest_label}"
                    elif v_type == 'v_bar':
                        label = f"Entity: {highest_label}; value: {highest_value}"
                    if is_multi:
                        label += f"; group: {annot_model['name']}"
                        ariaLabels.append(label)

            if len(entities) > 0:
                annot_json['tasks'].append({"question": q['query'], "labels": q_labels, "entity": entities, "aria-label": ariaLabels})

        if len(annot_json['tasks']) > 0:    
            data_entries = sortby_value(data_entries, v_type, is_multi) # sort h_bars
            unique = {(d["Entity"], d.get("group",""), d["value"]): d for d in data_entries}
            output_json['vconcat'][0]['data']['values'] = list(unique.values())
            output_json['name'] = filename
            save_chart_batch(output_json, annot_json, input_path, output_path, filename.strip('.json'))

def process_json(input_path: str, subset: str, output_path: str, base_path: str, is_multi: bool):
    Path.mkdir(Path(output_path), exist_ok=True)
    for i in trange(len(os.listdir(os.path.join(input_path, subset, 'annotations')))):
        filename = os.listdir(os.path.join(input_path, subset, 'annotations'))[i]
        if not filename.endswith(".json") or filename.startswith("two_col") or filename.startswith("multi_col"): continue
        f1 = open(os.path.join(input_path, subset, f'{subset}_human.json'), 'r')
        ques_json = json.load(f1)
        questions = [x for x in ques_json if x["imgname"]==filename.replace('.json', '.png')]
        if not questions: continue
        f2 = open(os.path.join(input_path, subset, 'annotations', filename), 'r')
        annot_json = json.load(f2)
        write_tasks(annot_json, questions, os.path.join(input_path,subset), base_path, output_path, filename, annot_json['type'], is_multi=is_multi)
        f1.close()
        f2.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA/")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA_VegaAltAir")
    parser.add_argument("--base_path", type=str, default="./data")
    parser.add_argument('--is_multi', action='store_true')
    args = vars(parser.parse_args())

    process_json(args['chartqa_path'], args['subset'], args['output_path'], args['base_path'], args['is_multi'])
