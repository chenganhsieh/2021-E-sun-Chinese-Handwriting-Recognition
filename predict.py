import torch
from PIL import Image


@torch.no_grad()
def predict(image: Image) -> str:
    prediction = '陳'

    return prediction
