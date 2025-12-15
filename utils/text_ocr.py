import json
import argparse
from PIL import Image
from pytesseract import pytesseract
pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def txt_loss(img, chart_json: dict) -> float:
    img_h = img.resize((int(img.size[0] / 2), int(img.size[1] / 2))) # resize to 1/2 to make sure text are still readable
    img_q = img.resize((int(img.size[0] / 4), int(img.size[1] / 4))) # resize to 1/4 to make sure text are still readable
    img_e = img.resize((int(img.size[0] / 8), int(img.size[1] / 8)))
    # img_s = img.resize((int(img.size[0] / 16), int(img.size[1] / 16)))
    ocr_h = pytesseract.image_to_string(img_h).lower()
    ocr_q = pytesseract.image_to_string(img_q).lower()
    ocr_e = pytesseract.image_to_string(img_e).lower()
    # ocr_s = pytesseract.image_to_string(img_s).lower()
    cnt = 0
    for entry in chart_json['vconcat'][0]['data']['values']:
        if str(entry['Entity']).lower() in ocr_h:
            cnt += 1
        if str(entry['value']).lower() in ocr_h:
            cnt += 1
        if str(entry['Entity']).lower() in ocr_q:
            cnt += 1
        if str(entry['value']).lower() in ocr_q:
            cnt += 1
        if str(entry['Entity']).lower() in ocr_e:
            cnt += 1
        if str(entry['value']).lower() in ocr_e:
            cnt += 1
        # if str(entry['Entity']).lower() in ocr_s:
        #     cnt += 1
        # if str(entry['value']).lower() in ocr_s:
        #     cnt += 1
    return cnt / (len(chart_json['vconcat'][0]['data']['values']) * 6)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="~/Dataset/ChartQA_VegaAltAir/optims/35422616009087.png")
    parser.add_argument("--chartjson_path", type=str, default="~/Dataset/ChartQA_VegaAltAir/vegas/35422616009087.json")
    args = vars(parser.parse_args())
    with open(args['chartjson_path'], encoding='utf-8') as f:
        print('Text OCR result:', txt_loss(Image.open(args['img_path']), json.load(f)))
