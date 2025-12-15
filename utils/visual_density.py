import os
import numpy as np
import argparse
from PIL import Image

# Visual Density is a metric that measures the area of inks used in the chart [0, 1]
def vd_loss(img_arr: np.ndarray) -> float:
    bg_ratio = img_arr[img_arr>253].size / img_arr.size
    M = 0.495859 # the average VD of ChartQA
    STD = 0.262958 #  the std VD of ChartQA
    if bg_ratio < M + STD and bg_ratio > M - STD:
        return 0
    return np.abs(bg_ratio - M)

# loss to aviod overlaps of bars
def overlap_loss(im_dict: dict, params: dict, is_multi: bool, is_pie: bool) -> float:
    if is_pie:
        if im_dict['vconcat'][0]['layer'][0]['mark']['innerRadius'] < im_dict['vconcat'][0]['layer'][1]['mark']['radius']:
            return 0
        return 1
    assert im_dict['vconcat'][0]['data']['values']
    assert im_dict['vconcat'][0]['layer'][0]['encoding']['size']['value']
    bar_num = len(im_dict['vconcat'][0]['data']['values'])
    if is_multi and int(params['x_agg']) == 1: # stacked bar chart
        bar_num /= 2
    if params['x_ot'] == 0:
        im_ll = im_dict['vconcat'][0]['height'] - 20
    else:
        im_ll = im_dict['vconcat'][0]['width']
    barwidth = im_dict['vconcat'][0]['layer'][0]['encoding']['size']['value']
    if bar_num * (barwidth + 1) > im_ll: # overlap occurs
        return 1
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="~/Dataset/ChartQA/train/png")
    args = vars(parser.parse_args())

    VDs = []
    for img_path in os.listdir(args['data_path']):
        if not img_path.endswith('.png'): continue
        image = Image.open(os.path.join(args['data_path'], img_path)).convert("RGB")
        gary_image = np.array(image.convert('L'))
        VDs.append(gary_image[gary_image>253].size / gary_image.size)
    print(np.mean(VDs), np.std(VDs))
    # 0.495859 0.262958
