import json
import torch
from torchvision import transforms
from torchvision.transforms.functional import pad
import numpy as np

model_path = './model_weight/resnet50.pth'
classes_path = './classes.json'
model = torch.load(model_path)
model.cuda()
model.eval()

with open(classes_path, 'r') as f:
    classes = json.load(f)

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
    y_pred = model(x.cuda())
    idx = np.argmax(y_pred.cpu().data.numpy(), axis=1)[0]
    char = classes[idx]
    print(idx)
    return char

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from PIL import Image
    image_name = 'pure_ysun/鼎/00099.jpg'
    image = Image.open(image_name)
    print(predict(image))

