import cv2
import numpy as np
from typing import List

# DEBUG: print losses
def debug(chart_json: dict, bboxes: List, im, params: dict, predictions: List, filepath: str):
    im_np = np.array(im)
    from utils.visual_density import vd_loss
    from utils.metrics import wave_metric
    from utils.text_ocr import txt_loss
    write_text(im_np, \
            # str(overlap_loss(chart_json, params, is_multi)) + ', ' \
            str(np.round(vd_loss(predictions[2]), 2)) + ', t=' \
            + str(np.round(txt_loss(predictions[1], chart_json), 2)) + ', c=' \
            + str(np.round(wave_metric(im), 2)))
    if len(bboxes) > 0:
        for bb in bboxes:
            bbox = bb['bbox']
            cv2.rectangle(im_np,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0, 255, 0), 2)
    cv2.imwrite(filepath, im_np)

def write_text(im: np.ndarray, text: str):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20,20)
    fontScale              = 1
    fontColor              = (0,0,0)
    thickness              = 1
    lineType               = 2

    cv2.putText(im, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
