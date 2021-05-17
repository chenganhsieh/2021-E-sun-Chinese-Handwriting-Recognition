
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

def macro_f1(num_class, y_pred, y_true):
    tps , fps, fns = [0]*num_class, [0]*num_class, [0]*num_class
    num_samples, f1_score = len(y_pred), 0
    for i in range(num_samples):
        y_pred_i = y_pred[i]
        y_true_i = y_true[i]
        if y_pred_i == y_true_i:
            tps[y_pred_i] += 1
        else:
            fps[y_pred_i] += 1
            fns[y_true_i] += 1
    
    for i in range(num_class):
        if tps[i]:
            tp, fp, fn = tps[i], fps[i], fns[i]
            p ,r = tp/(tp + fp), tp/(tp + fn)
            f1_score += 2*p*r/(p+r)
    return sum(tps)/num_samples, f1_score/num_class

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN model for Chinese Handwriting Recognition.")
    
    #Directories
    parser.add_argument("--data_dir", type=str, default='./clean', help="The directory which contains the training data.")
    parser.add_argument("--model_save_path", type=str, default='./model_weight/resnext101_32x8d.pth', help="Where to store the final model.")
    
    #DataAugumentation
    parser.add_argument('--crop_lower_bound', type=float, default=0.8, help="Parameters for RandomResizedCrop().")
    parser.add_argument('--color_jitter', type=float, default=0.25, help="Parameters for ColorJitter().")

    #Training
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    
    parser.add_argument("--seed", type=int, default=390625, help="A seed for reproducible training.")
    parser.add_argument("--pretrained_weight", type=bool, default=False, help="Whether to use pretrained weight provided in pytorch or not.")
    parser.add_argument('--num_workers', type=int, default=16, help="num of workers for dataloader.")
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
        transforms.RandomResizedCrop(224, scale=(args.crop_lower_bound, 1)),
        transforms.RandomApply([transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter, saturation=args.color_jitter, hue=args.color_jitter)],p=0.3),
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
    # exit()

    class_to_idx = dataset.class_to_idx
    num_class = len(class_to_idx)
    if args.debug: # Cut dataset size in debug mode
        dataset,_ = random_split(dataset,[100,len(dataset)-100])
    num_train_sample = int(len(dataset)*0.85)
    num_eval_sample = len(dataset) - num_train_sample
    
    # Prepare dataloader 
    print(f'Training with {num_train_sample} train_data, {num_eval_sample} eval_data across {num_class} classes.')
    train_dataset, eval_dataset = random_split(dataset, [num_train_sample, num_eval_sample])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    num_train_batch = len(train_loader)
    num_eval_batch = len(eval_loader)
    
    # Load model
    if args.pretrained_weight: # If using pretrained weight, modify fully-connected layer output to num_class.
        model = models.resnext101_32x8d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)
    else:
        model = models.resnext101_32x8d(pretrained=False, num_classes=num_class)
    model.cuda()
    if not args.debug:
        wandb.watch(model)   
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for epoch in range(args.num_train_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()  
        epoch_start_time = time.time()
        train_loss, eval_loss= 0.0, 0.0
        y_pred_list , y_true_list= [], []
        for i, (x,y) in enumerate(train_loader):
            y_pred = model(x.cuda())
            loss = criterion(y_pred, y.cuda())
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (i+1) % args.gradient_accumulation_steps == 0 or (i+1) == num_train_batch:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item()
            y_pred_list.extend(np.argmax(y_pred.cpu().data.numpy(), axis=1).tolist())
            y_true_list.extend(y.tolist())
            
            print(f'[{i:03d}/{num_train_batch}]', end='\r')
        train_acc, train_f1 = macro_f1(num_class, y_pred_list, y_true_list)
        # Validation step
        model.eval()
        y_pred_list , y_true_list= [], []
        with torch.no_grad():
            for i, (x,y) in enumerate(eval_loader):
                y_pred = model(x.cuda())
                loss = criterion(y_pred, y.cuda())
                eval_loss += loss.item()
                y_pred_list.extend(np.argmax(y_pred.cpu().data.numpy(), axis=1).tolist())
                y_true_list.extend(y.tolist())
                print(f'[{i:03d}/{num_eval_batch}]', end='\r')
        eval_acc, eval_f1 = macro_f1(num_class, y_pred_list, y_true_list)
        # Summarize per-epoch result
        train_loss /= num_train_sample
        eval_loss /= num_eval_sample
        print(f'epoch [{epoch+1:02d}/{args.num_train_epochs}]: {time.time()-epoch_start_time:.2f} sec(s)')
        print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, train macro_f1: {train_f1:.4f}')
        print(f' eval loss: {eval_loss:.4f},  eval acc: {eval_acc:.4f},  eval macro_f1: {eval_f1:.4f}')
        if not args.debug:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "eval_loss": eval_loss, "eval_acc": eval_acc, "train macro_f1": train_f1, "eval macro_f1": eval_f1})
        torch.save(model, args.model_save_path)  
    return

if __name__ == "__main__":
    args = parse_args()
    if not args.debug:
        wandb.init(project='chinese_handwriting_recognition', entity='waste30minfornaming',name='resnext101_32x8d')
        config = wandb.config
        config.update(args)
    set_seed(args.seed)
    main(args)