import json
import torch
from torchvision import transforms
from torchvision.transforms.functional import pad
import numpy as np


model_list = []
for i in range(1, 8 + 1):
    model = torch.load(f'ckpt/ensemble_b5_{i}.pth')
    model.cuda()
    model.eval()
    model_list.append(model)
classes_path = 'ckpt/classes4839.json'
in_class_dict_path = 'ckpt/class_to_idx.json'

with open(classes_path, 'r') as f:
    classes = json.load(f)
with open(in_class_dict_path, 'r') as f:
    in_class_dict = json.load(f)

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        if h > w:
            hp = (h - w) // 2
            padding = (hp,0,h-w-hp,0)
        else:
            vp = (w - h) // 2
            padding = (0, vp, 0, w-h-vp)

        return pad(image, padding, (255,255,255), 'constant')

test_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def image_loader(transforms, image):
    image = transforms(image).float()
    # image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def predict(image):
    x = image_loader(test_transform, image).cuda()
    ens_logits = None
    isnull_vote = 0
    char_rec = []
    for model in model_list:
        logits = model(x)
        if ens_logits is None:
            ens_logits = logits
        else:
            ens_logits += logits
        char = classes[torch.argmax(logits, dim=1)[0]]
        if char not in in_class_dict:
            isnull_vote += 1
            char = f'{char}(isnull)'
        char_rec.append(char)
    if isnull_vote > len(model_list) / 2:
        char = 'isnull'
    else:
        char = classes[torch.argmax(ens_logits, dim=1)[0]]
        if char not in in_class_dict:
            char = 'isnull'
    # print(char_rec)
    return char

def predict_batch(batch, labels):
    batch = batch.cuda()
    ens_logits = None
    ens_isnull = None
    for model in model_list:
        logits = model(batch)
        if ens_logits is None:
            ens_logits = logits
        else:
            ens_logits += logits
        indices = torch.argmax(logits, dim=1)
        isnull = torch.tensor([classes[idx] not in in_class_dict for idx in indices], dtype=torch.int)
        if ens_isnull is None:
            ens_isnull = isnull
        else:
            ens_isnull += isnull

    indices = torch.argmax(ens_logits, dim=1)
    corrects = 0
    for idx, isnull, label in zip(indices, ens_isnull, labels):
        # if isnull > len(model_list) / 2:
        if isnull >=4:
            pred = "isnull"
        else:
            if classes[idx] not in in_class_dict:
                pred = "isnull"
            else:
                pred = classes[idx]
        if classes[label] not in in_class_dict:
            label = "isnull"
        else:
            label = classes[label]
        corrects += (pred == label)
        if pred != label:
            print(f"gt: {label}, pred: {pred}")
    return corrects


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from PIL import Image
    image_name = 'pure_ysun/鼎/00099.jpg'
    image = Image.open(image_name)
    print(predict(image))

