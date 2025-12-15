import os
import argparse
import json
from pathlib import Path

def _extract_first_unit_spec(spec):
    """
    Internal helper: Extract the first real unit spec from vconcat or layer.
    Returns the unit spec (dict).
    """
    if "vconcat" in spec:
        return spec["vconcat"][0]
    return spec

def get_field_by_type(enc, channel, vega_type):
    if channel in enc and isinstance(enc[channel], dict):
        if enc[channel].get("type") == vega_type:
            return enc[channel].get("field")
    return None

def convert_bar_to_pie(bar_spec):
    """
    Convert a Vega-Lite bar chart specification (possibly layered or vconcat)
    into a clean pie-chart specification.
    """
    unit = _extract_first_unit_spec(bar_spec)

    # Extract data
    data = unit.get("data", {})

    # Extract encodings (from bar layer if needed)
    encoding = unit.get("encoding", {})
    if "layer" in unit:
        # Take encoding from the bar layer (the first layer)
        bar_layer = unit["layer"][0]
        encoding = bar_layer.get("encoding", encoding)
    pie = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "vconcat": [
            {
                "width": unit.get("width", 600),
                "height": unit.get("height", 600),
                "data": data,
                "layer": [
                    {
                        "mark": {"type": "arc", "outerRadius": 200, "innerRadius": 0},
                        "encoding": {
                            "theta": {"field": "value", "type": "quantitative", "stack": True},
                            "color": {"field": "Entity", "type": "nominal", "sort": None, "legend": {
                                "title": None,
                                "labelFontSize": 24,
                                "labelLimit": 0
                            },
                            "scale": {"range": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f" , "#bcbd22", "#17becf"]}
                            },
                            "order": {
                                "field": "value",
                                "sort": "descending"
                            }
                        }
                    },
                    {
                        "mark": {"type": "text", "radius": 100, "fontSize": 14, "align": "center", "baseline": "middle"},
                        "encoding": {
                            "theta": {"field": "value", "type": "quantitative", "stack": True},
                            "text": {"field": "value", "type": "quantitative"},
                            "order": {
                                "field": "value",
                                "sort": "descending"
                            }
                        }
                    }
                ]
            }
        ]
    }
    return pie

def convert_pie_to_bar(spec, orientation="vertical"):
    # Extract unit spec
    unit = spec["vconcat"][0] if "vconcat" in spec else spec

    enc = unit.get("encoding", {})
    theta = enc.get("theta", {})
    color = enc.get("color", {})

    value_field = theta.get("field")
    category_field = color.get("field")

    if value_field is None or category_field is None:
        raise ValueError(
            f"This is not a valid pie chart.\nEncoding={enc}"
        )

    if orientation == "horizontal":
        bar_enc = {
            "y": {"field": category_field, "type": "nominal"},
            "x": {"field": value_field, "type": "quantitative"},
            "color": {"field": category_field, "type": "nominal"}
        }
    else:
        bar_enc = {
            "x": {"field": category_field, "type": "nominal"},
            "y": {"field": value_field, "type": "quantitative"},
            "color": {"field": category_field, "type": "nominal"}
        }

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": unit.get("width", 400),
        "height": unit.get("height", 400),
        "data": unit.get("data", {}),
        "mark": "bar",
        "encoding": bar_enc
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA_VegaAltAir/vegas_userstudy")
    parser.add_argument("--output_path", type=str, default="/netpool/homes/wangyo/Dataset/ChartQA_VegaAltAir/vegas_userstudy_pie")
    parser.add_argument("--to_bar", action='store_true')
    parser.add_argument("--orientation", type=str, default="v_bar")
    args = vars(parser.parse_args())

    Path.mkdir(Path(args['output_path']), exist_ok=True)

    
    for data_json in os.listdir(args['input_path']):
        if not data_json.endswith('.json'): continue
        with open(os.path.join(args['input_path'], data_json), 'r') as f:
            chart_json = json.load(f)

        if args['to_bar']:
            converted_json = convert_pie_to_bar(chart_json, orientation=args['orientation'])
        else:
            converted_json = convert_bar_to_pie(chart_json)

        with open(os.path.join(args['output_path'], data_json), 'w') as f:
            json.dump(converted_json, f, indent=4)
