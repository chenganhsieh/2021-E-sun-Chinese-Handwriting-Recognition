
import argparse
from numpy.core.fromnumeric import resize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms.functional import pad
import torchvision.models as models
import numpy as np
import random
import wandb
import time

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN model for Chinese Handwriting Recognition.")
    
    #Directories
    parser.add_argument("--data_dir", type=str, default='./images', help="The directory which contains the training data.")
    parser.add_argument("--model_save_path", type=str, default='model_transforms.pth', help="Where to store the final model.")
    
    #Training
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the dataloader.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    
    
    parser.add_argument("--seed", type=int, default=390625, help="A seed for reproducible training.")
    parser.add_argument("--pretrained_weight", type=bool, default=False, help="Whether to use pretrained weight provided in pytorch or not.")
    parser.add_argument("--debug", action="store_true", help="Activate debug mode and run training only with a subset of data.")
    
    args = parser.parse_args()
    return args


def main(args):
    
    # Prepare dataset with specified transform 
    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            l = max(w,h)
            hp = int((l - w) / 2)
            vp = int((l - h) / 2)
            padding = (hp, vp, l-w-hp, l-h-vp)
            return pad(image, padding, (255,255,255), 'constant')
    
    train_transform = transforms.Compose([
        SquarePad(),
        transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomRotation(15,fill=(255,255,255)),
        transforms.ToTensor(),
    ])
    # test_transform = transforms.Compose([
    #     SquarePad(),
    #     transforms.ToTensor(),
    # ])
    dataset = datasets.ImageFolder(args.data_dir, transform = train_transform)
    
    # Output a training image for observation
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png',np.transpose(dataset[0][0].numpy(),(1,2,0)))
    
    class_to_idx = dataset.class_to_idx
    num_class = len(class_to_idx)
    if args.debug: # Cut dataset size in debug mode
        dataset,_ = random_split(dataset,[100,len(dataset)-100])
    num_train_sample = int(len(dataset)*0.85)
    num_eval_sample = len(dataset) - num_train_sample
    
    # Prepare dataloader 
    print(f'Training with {num_train_sample} train_data, {num_eval_sample} eval_data across {num_class} classes.')
    train_dataset, eval_dataset = random_split(dataset, [num_train_sample, num_eval_sample])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    num_train_batch = len(train_loader)
    num_eval_batch = len(eval_loader)
    
    # Load model
    if args.pretrained_weight: # If using pretrained weight, modify fully-connected layer output to num_class.
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)
    else:
        model = models.resnet50(pretrained=False, num_classes=num_class)
    model.cuda()
    if not args.debug:
        wandb.watch(model)   
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for epoch in range(args.num_train_epochs):
        # Training step
        model.train()
        epoch_start_time = time.time()
        train_acc, train_loss, eval_acc, eval_loss= 0.0, 0.0, 0.0, 0.0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x.cuda())
            loss = criterion(y_pred, y.cuda())
            loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(y_pred.cpu().data.numpy(), axis=1) == y.numpy())
            train_loss += loss.item()
            print(f'[{i:03d}/{num_train_batch}]', end='\r')
        
        # Validation step
        model.eval()
        with torch.no_grad():
            for i, (x,y) in enumerate(eval_loader):
                y_pred = model(x.cuda())
                loss = criterion(y_pred, y.cuda())
                eval_acc += np.sum(np.argmax(y_pred.cpu().data.numpy(), axis=1) == y.numpy())
                eval_loss += loss.item()
                print(f'[{i:03d}/{num_eval_batch}]', end='\r')
        
        # Summarize per-epoch result
        train_acc /= num_train_sample
        train_loss /= num_train_sample
        eval_acc /= num_eval_sample
        eval_loss /= num_eval_sample
        print(f'epoch [{epoch+1:02d}/{args.num_train_epochs}]: {time.time()-epoch_start_time:.2f} sec(s)')
        print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        print(f' eval loss: {eval_loss:.4f},  eval acc: {eval_acc:.4f}')
        if not args.debug:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "eval_loss": eval_loss, "eval_acc": eval_acc})
        torch.save(model, args.model_save_path)  
    return

if __name__ == "__main__":
    args = parse_args()
    if not args.debug:
        wandb.init(project='chinese_handwriting_recognition', entity='guan27',name='transforms')
        config = wandb.config
        config.update(args)
    set_seed(args.seed)
    main(args)