import json
import torch
from torchvision import transforms
from torchvision.transforms.functional import pad
import numpy as np

model_path = '/tmp/minghao/ensemble_b5_1.pth'
classes_path = '/tmp/minghao/classes4839.json'
in_class_dict_path = '/tmp/minghao/class_to_idx.json'
model = torch.load(model_path)
model.cuda()
model.eval()

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
    x = image_loader(test_transform, image)
    # y_pred = model(x.cuda())
    y_pred = torch.nn.functional.softmax(model(x.cuda()).squeeze(), dim=0)
    # idx = np.argmax(y_pred.cpu().data.numpy(), axis=1)[0]
    idx = torch.argmax(y_pred)
    # print(round(y_pred[idx].item(), 2))
    # print(round(y_pred[0].item(), 2))
    char = classes[idx]
    # print(char)
    if char not in in_class_dict:
        char = 'isnull'
    return char

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from PIL import Image
    image_names = [
        'selected_imgs/isnull/0525_180336_d789.jpg',
        'selected_imgs/女/0525_180317_87ef.jpg',
        'selected_imgs/峻/0525_180817_18de.jpg',
        'selected_imgs/營/0525_180838_00e7.jpg',
        'selected_imgs/馬/0525_180405_97f2.jpg',
    ]
    # image_name = 'selected_imgs/0525_180325_53dd.jpg'
    # image_name = 'selected_imgs/0525_180352_b4ea.jpg'
    # image_name = 'selected_imgs/0525_182356_15c9.jpg'
    # image_name = 'selected_imgs/0526_100509_417f.jpg'
    # image_name = 'selected_imgs/0526_100630_e2b2.jpg'
    # image_name = 'selected_imgs/00000.jpg'
    # image_name = 'selected_imgs/00006.jpg'
    # image_name = 'selected_imgs/00007.jpg'
    
    for image_name in image_names:
        image = Image.open(image_name)
        print(predict(image))

