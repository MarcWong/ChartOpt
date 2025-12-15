import json
import os
import shutil
import numpy as np
import altair as alt
from pathlib import Path
from matplotlib import colors as mcolors
from svg.path import parse_path
from xml.dom import minidom
from typing import List
from PIL import Image
import colorsys
from utils.debug import debug
from io import BytesIO, StringIO

def init_bo_params(is_multi: bool, is_pie: bool) -> List:
    parameters = []
    if is_pie: # pie chart
    # x0: inner_radius_ratio
    # x1: radius_mark
    # x2: font_size_mark
    # x3, x4, x5: highlight color (h, s, v)
        for i in range(6):
            parameters.append(
                {
                "name": f"x{i}",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float"
            })
        # is pie
        parameters.append({
            "name": "x_pi",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "int"
        })
    else: # bar chart
    # x0: aspect_ratio
    # x1: font_size_axis
    # x2: font_size_mark
    # x3: bar_size (bandwidth)
    # x4, x5, x6: highlight color (h, s, v)
    # x7, x8, x9: background color (h, s, v)
        for i in range(10):
            parameters.append(
                {
                "name": f"x{i}",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float"
            })
        # v_bar label rotation
        parameters.append({
            "name": "x_rt",
            "type": "range",
            "bounds": [0.0, 2.0],
            "value_type": "int"
        })
        # bar orientation
        parameters.append({
            "name": "x_ot",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "int"
        })
        if is_multi:
            # bar aggregation, only for multigroup bars
            parameters.append({
                "name": "x_agg",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "int"
            })
    return parameters

def calc_param(dict_key: str, param: float, is_discrete: bool = False) -> float:    
    PARAM_BOUNDS = {
        'aspect_ratio': [0.33, 3],
        'font_size_axis': [10, 36],
        'font_size_mark': [10, 36],
        'bar_size': [20, 180],
        'axis_label_rotation': [-90, -45, 0],
    }
    if is_discrete:
        return PARAM_BOUNDS[dict_key][int(param)]
    return PARAM_BOUNDS[dict_key][0] + (PARAM_BOUNDS[dict_key][1] - PARAM_BOUNDS[dict_key][0]) * param

def change_orientation(chart_json: dict, annotation: dict, ot: int, is_multi: bool = False):
    # ot==0: horizontal, 1: vertical
    if ot == 0 and annotation['type'] == 'v_bar' or ot == 1 and annotation['type'] == 'h_bar':
        tmp = chart_json['vconcat'][0]['encoding']['x']
        chart_json['vconcat'][0]['encoding']['x'] = chart_json['vconcat'][0]['encoding']['y']
        chart_json['vconcat'][0]['encoding']['y'] = tmp
        for i, task in enumerate(annotation['tasks']):
            for j, label in enumerate(task['aria-label']):
                parts = label.split(';')
                alabel = f"{parts[1].strip()}; {parts[0].strip()}"
                if is_multi:
                    alabel += f"; {parts[2].strip()}"
                annotation['tasks'][i]['aria-label'][j] = alabel
    if ot == 0 and annotation['type'] == 'v_bar': # v_bar -> h_bar
        annotation['type'] = 'h_bar'
        chart_json['vconcat'][0]['layer'][1]['encoding']['x'] = chart_json['vconcat'][0]['layer'][1]['encoding']['y']
        safely_delete_nested_key(chart_json, ['vconcat', 0, 'layer', 1, 'encoding', 'y'])
    elif ot == 1 and annotation['type'] == 'h_bar':   # h_bar -> v_bar
        annotation['type'] = 'v_bar'
        chart_json['vconcat'][0]['layer'][1]['encoding']['y'] = chart_json['vconcat'][0]['layer'][1]['encoding']['x']
        safely_delete_nested_key(chart_json, ['vconcat', 0, 'layer', 1, 'encoding', 'x'])

    return chart_json, annotation

def safely_delete_nested_key(data, path_parts):
    """
    Safely traverses a dictionary/list structure and deletes a nested key.
    
    :param data: The starting dictionary/list.
    :param path_parts: A list of keys/indices representing the path to the element 
                       whose key needs to be deleted (the last element).
    """

    # The key to delete is the last element of the path
    key_to_delete = path_parts[-1]
    # The container is everything before the key to delete
    container_path = path_parts[:-1]

    current = data
    for part in container_path:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and isinstance(part, int) and part < len(current):
            current = current[part]
        else:
            return

    if isinstance(current, dict) and key_to_delete in current:
        del current[key_to_delete]
    return current

def update_chart(CHART_JSON: dict, params: dict, annotation: dict, query: dict, datapath: str = 'data', filename: str = 'chart', is_multi: bool = False, is_pie: bool = False, predictions = None):
    if is_pie:
        chart_json = CHART_JSON.copy()
        if params['x_pi'] == 1:
            chart_json['vconcat'][0]['layer'][0]['mark']['innerRadius'] = 0
        else:
            chart_json['vconcat'][0]['layer'][0]['mark']['innerRadius'] = params['x0'] * chart_json['vconcat'][0]['layer'][0]['mark']['outerRadius']
        chart_json['vconcat'][0]['layer'][1]['mark']['radius'] = params['x1'] * chart_json['vconcat'][0]['layer'][0]['mark']['outerRadius']
        chart_json['vconcat'][0]['layer'][1]['mark']['fontSize'] = calc_param('font_size_mark', params['x2'])
        hl_color_rgb = mcolors.to_hex(colorsys.hsv_to_rgb(params['x3'], params['x4'], params['x5']))
        for i, dd in enumerate(chart_json['vconcat'][0]['data']['values']):
            if dd['Entity'] == query['entity'][0]:
                chart_json['vconcat'][0]['layer'][0]['encoding']['color']['scale']['range'][i] = hl_color_rgb
                break
    else:
        chart_json, annotation = change_orientation(CHART_JSON, annotation, params['x_ot'], is_multi=is_multi)
        chart_json['vconcat'][0]['width'] = chart_json['vconcat'][0]['height'] * calc_param('aspect_ratio', params['x0'])
        chart_json['vconcat'][0]['layer'][1]['mark']['fontSize'] = calc_param('font_size_mark', params['x2'])
        chart_json['vconcat'][0]['layer'][0]['encoding']['size']['value'] = calc_param('bar_size', params['x3'])

        hl_color_rgb = mcolors.to_hex(colorsys.hsv_to_rgb(params['x4'], params['x5'], params['x6']))
        bg_color_rgb = mcolors.to_hex(colorsys.hsv_to_rgb(params['x7'], params['x8'], params['x9']))
        if not is_multi: # find the highlight color conditions
            chart_json['vconcat'][0]['layer'][0]['encoding']['color']['value'] = bg_color_rgb
            for _, entity in enumerate(query['entity']):
                f = False
                for dd in chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition']:
                    if dd['test'] == f"datum.Entity === '{entity}'":
                        dd['value'] = hl_color_rgb
                        f = True
                        break
                if f: continue
                chart_json['vconcat'][0]['layer'][0]['encoding']['color']['condition'].append({
                    "test": f"datum.Entity === '{entity}'",
                    "value": hl_color_rgb
                })
        else: # multi-grouped bar chart, now support 2 groups of colors
            chart_json['vconcat'][0]['encoding']['color']['scale'] = {'range': [hl_color_rgb, bg_color_rgb]}
            safely_delete_nested_key(chart_json, ['vconcat', 0, 'encoding', 'xOffset'])
            safely_delete_nested_key(chart_json, ['vconcat', 0, 'encoding', 'yOffset'])
            safely_delete_nested_key(chart_json, ['vconcat', 0, 'layer', 1, 'encoding', 'x', 'stack'])
            safely_delete_nested_key(chart_json, ['vconcat', 0, 'layer', 1, 'encoding', 'y', 'stack'])

            main_enc = chart_json['vconcat'][0]['encoding']
            text_enc = chart_json['vconcat'][0]['layer'][1]['encoding']

            # CASE 1: NOT AGGREGATED → GROUPED BARS WITH OFFSETS
            if int(params['x_agg']) == 0:
                if annotation['type'] == 'h_bar':
                    main_enc['yOffset'] = {'sort': None, "field": "group"}
                    text_enc['yOffset'] = main_enc['yOffset']
                    safely_delete_nested_key(text_enc, ['xOffset'])
                else:
                    main_enc['xOffset'] = {'sort': None, "field": "group"}
                    text_enc['xOffset'] = main_enc['xOffset']
                    safely_delete_nested_key(text_enc, ['yOffset'])
            # CASE 2: For stacked charts, text must stack consistently
            else:
                if annotation['type'] == 'h_bar':
                    chart_json['vconcat'][0]['layer'][1]['encoding']['x']['stack'] = 'zero'
                else:
                    chart_json['vconcat'][0]['layer'][1]['encoding']['y']['stack'] = 'zero'

        if annotation['type'] == 'h_bar':
            chart_json['vconcat'][0]['encoding']['y']['axis']['labelFontSize'] = calc_param('font_size_axis', params['x1'])
            chart_json['vconcat'][0]['layer'][1]['mark']['xOffset'] = calc_param('font_size_axis', params['x1'])/2
            chart_json['vconcat'][0]['layer'][1]['mark']['yOffset'] = 0
            if is_multi:
                chart_json['vconcat'][0]['layer'][1]['mark']['dx'] = calc_param('font_size_axis', params['x1'])
                chart_json['vconcat'][0]['layer'][1]['mark']['align'] = 'center'
            else:
                chart_json['vconcat'][0]['layer'][1]['mark']['dx'] = 16
                chart_json['vconcat'][0]['layer'][1]['mark']['align'] = 'left'
            chart_json['vconcat'][0]['encoding']['y']['axis']['labelAngle'] = 0
        elif annotation['type'] == 'v_bar':
            chart_json['vconcat'][0]['encoding']['x']['axis']['labelFontSize'] = calc_param('font_size_axis', params['x1'])
            chart_json['vconcat'][0]['layer'][1]['mark']['xOffset'] = 0
            chart_json['vconcat'][0]['layer'][1]['mark']['yOffset'] = -calc_param('font_size_axis', params['x1'])*2/3
            chart_json['vconcat'][0]['layer'][1]['mark']['dx'] = 0
            chart_json['vconcat'][0]['layer'][1]['mark']['align'] = 'center'
            chart_json['vconcat'][0]['encoding']['x']['axis']['labelAngle'] = calc_param('axis_label_rotation', params['x_rt'], is_discrete=True)

    chart = alt.Chart.from_json(json.dumps(chart_json))

    png_buffer = BytesIO()
    chart.save(png_buffer, format='png')
    png_buffer.seek(0)
    svg_buffer = StringIO()
    chart.save(svg_buffer, format='svg')
    svg_buffer.seek(0)

    im = Image.open(png_buffer).convert("RGB")
    im_np = np.array(im)

    bboxes = get_bboxes(svg_buffer, annotation, query, np.shape(im_np), is_pie=is_pie)

    if not datapath == 'data':
        chart.save(f'{datapath}/{filename}.png')
        with open(f'{datapath}/{filename}.json', 'w') as out_file:
            json.dump(chart_json, out_file)

    # if predictions is not None:
    #     debug(chart_json, bboxes, im, params, predictions, f'{datapath}/{filename}_bbox.png')
    return chart_json, bboxes, png_buffer

def save_chart_batch(chart_json: dict, annotation: dict, input_path: str, output_path: str, filename: str = 'chart'):
    Path.mkdir(Path(os.path.join(output_path, 'svgs')), exist_ok=True)
    Path.mkdir(Path(os.path.join(output_path, 'vegas')), exist_ok=True)
    Path.mkdir(Path(os.path.join(output_path, 'annotations')), exist_ok=True)
    Path.mkdir(Path(os.path.join(output_path, 'png')), exist_ok=True)
    Path.mkdir(Path(os.path.join(output_path, 'tables')), exist_ok=True)

    chart = alt.Chart.from_json(json.dumps(chart_json))
    chart.save(os.path.join(output_path, 'svgs', f'{filename}.svg'))
    with open(os.path.join(output_path, 'vegas', f'{filename}.json'), 'w') as out_file:
        json.dump(chart_json, out_file)
    with open(os.path.join(output_path, 'annotations', f'{filename}.json'), 'w') as out_file:
        json.dump(annotation, out_file)
    shutil.copy(os.path.join(input_path, 'png', filename+'.png'), os.path.join(output_path, 'png', filename+'.png'))
    shutil.copy(os.path.join(input_path, 'tables', filename+'.csv'), os.path.join(output_path, 'tables', filename+'.csv'))

def get_bboxes(svg_file, annotation:dict, query: dict, imshape: np.ndarray, is_pie: bool = False) -> List[np.ndarray]:
    xmldoc = minidom.parse(svg_file)
    PNT = xmldoc.getElementsByTagName("path")
    GROUP = xmldoc.getElementsByTagName("g")
    x_offset = 0
    if not is_pie:
        for g in GROUP:
            if g.getAttribute('class') == "mark-text role-axis-title":
                child = g.firstChild
                # find Y-axis label in h_bar or X-axis label in v_bar
                if (annotation['type'] == 'h_bar' and 'rotate(-90)' in child.getAttribute('transform')) \
                or (annotation['type'] == 'v_bar' and not 'rotate(-90)' in child.getAttribute('transform')):
                    x_offset = float(child.getAttribute('transform').strip('translate(-').split(',')[0]) + float(child.getAttribute('font-size')[0:1]) # offset caused by axis labels
                    # print(x_offset)
                    break

    bboxes = []
    for ariaLabel in query['aria-label']:
        for element in PNT:
            if element.getAttribute('aria-label') == ariaLabel:
                path_string = element.getAttribute('d')
                path = parse_path(path_string)
                bbox = path.boundingbox()
                if is_pie:
                    pie_offset_x = 0
                    pie_offset_y = 0

                    import re
                    # Accumulate any translate transforms in the hierarchy
                    parent = element.parentNode
                    while parent and parent.nodeType == parent.ELEMENT_NODE:
                        transform = parent.getAttribute('transform')
                        if transform and 'translate' in transform:
                            match = re.search(r'translate\(([^,]+),\s*([^)]+)\)', transform)
                            if match:
                                pie_offset_x += float(match.group(1))
                                pie_offset_y += float(match.group(2))
                        parent = parent.parentNode
                    
                    # Find the mark-arc group to get the container dimensions
                    svg_root = xmldoc.getElementsByTagName('svg')[0] if xmldoc.getElementsByTagName('svg') else None
                    if svg_root:
                        viewBox = svg_root.getAttribute('viewBox')
                        if viewBox:
                            parts = viewBox.split()
                            if len(parts) == 4:
                                width = float(parts[2])
                                height = float(parts[3])
                                # The pie center should be at chartWidth/2
                                chart_width = 621
                                pie_offset_x += chart_width / 2
                                pie_offset_y += height / 2
                    bbox[0] += pie_offset_x
                    bbox[2] += pie_offset_x
                    bbox[1] += pie_offset_y
                    bbox[3] += pie_offset_y
                else:
                    if annotation['type'] == 'h_bar':
                        bbox[2] += (2 * x_offset)
                    else:
                        bbox[1] += (-50)
                        bbox[3] += (x_offset)
                bbox[0] = max(0, bbox[0])
                bbox[1] = min(imshape[1] - 1, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = min(imshape[0] - 1, bbox[3])
                bboxes.append(
                    {
                        "label": ariaLabel,
                        "bbox": np.asarray(bbox, dtype=int)
                    }
                )
                break

    xmldoc.unlink()
    return bboxes

def load_json(data_path: str, annot_path: str):
    f = open(data_path, 'r', encoding='utf-8')
    f2 = open(annot_path, 'r', encoding='utf-8')
    # chart, annot
    return json.load(f), json.load(f2)
