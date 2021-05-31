import json
import torch
from torchvision import transforms
from torchvision.transforms.functional import pad
import numpy as np

model_path_1 = './model_weight/resnext50_32x4d_ensemble_1.pth'
model_path_2 = './model_weight/resnext50_32x4d_ensemble_2.pth'
model_path_3 = './model_weight/resnext50_32x4d_ensemble_3.pth'
classes_path = './classes4839.json'
in_class_dict_path = './class_to_idx.json'
model_1 = torch.load(model_path_1)
model_1.cuda()
model_1.eval()
model_2 = torch.load(model_path_2)
model_2.cuda()
model_2.eval()
model_3 = torch.load(model_path_3)
model_3.cuda()
model_3.eval()

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
    ])

def image_loader(transforms, image):
    image = transforms(image).float()
    # image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def predict(image):
    x = image_loader(test_transform, image)
    y_pred_1 = model_1(x.cuda())
    y_pred_2 = model_2(x.cuda())
    y_pred_3 = model_3(x.cuda())
    y_pred = y_pred_1 + y_pred_2 + y_pred_3
    idx = np.argmax(y_pred.cpu().data.numpy(), axis=1)[0]
    char = classes[idx]
    if char not in in_class_dict:
        char = 'isnull'
    return char

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from PIL import Image
    image_name = 'pure_ysun/鼎/00099.jpg'
    image = Image.open(image_name)
    print(predict(image))

