import torch
from typing import List
import numpy as np
from PIL import Image

DEVICE = 'cuda'

def init_network():
    from models.model import SalFormer
    from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel

    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = SalFormer(vit, bert).to(DEVICE)
    checkpoint = torch.load('./models/model_lr6e-5_wd1e-4.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, image_processor, tokenizer

def predict(model, image_processor, tokenizer, png_buffer, ques: str) -> List:
    """
    Execute the prediction.

    Args:
        png_buffer: a buffer containing the image data
        ques: a question string to feed into VisSalFormer
    Returns: [list]
        - heatmap from VisSalFormer (np.array)
        - Average WAVE score across pixels (float, [0, 1))
    """
    image = Image.open(png_buffer).convert("RGB")
    img_pt = image_processor(image, return_tensors="pt").to(DEVICE)
    inputs = tokenizer(ques, return_tensors="pt").to(DEVICE)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (mask * 255).astype(np.uint8)
    im_grey = image.convert('L')

    heatmap = np.resize(heatmap, (image.size[1], image.size[0]))
    return [heatmap, image, np.array(im_grey)]
