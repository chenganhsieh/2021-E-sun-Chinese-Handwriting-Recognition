
import argparse
import logging
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import time
import tqdm

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
    parser.add_argument("--data_dir", type=str, default='./images', help="The directory which contains the training data.")

    parser.add_argument(
        "--pretrained_weight",
        type=bool,
        help="Whether to use pretrained weight provided in pytorch.",
        default=False,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for the dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument("--model_save_path", type=str, default='model.pth', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=390625, help="A seed for reproducible training.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    args = parser.parse_args()
    return args

def main(args, logger):
    
    logger.info('hello world')
    set_seed(args.seed)

    train_transform = transforms.Compose([
        transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    # test_transform = transforms.Compose([
    #     transforms.CenterCrop(256),
    #     transforms.ToTensor(),
    # ])


    dataset = datasets.ImageFolder(args.data_dir, transform = train_transform)
    class_to_idx = dataset.class_to_idx
    num_class = len(class_to_idx)

    if args.debug:
        dataset,_ = random_split(dataset,[1000,len(dataset)-1000])
    train_size = int(len(dataset)*0.85)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    num_train_batch = len(train_loader)
    num_eval_batch = len(eval_loader)
    # model = models.resnet50(pretrained=args.pretrained_weight, num_classes=num_class).cuda()
    
    model = models.resnet50(pretrained=args.pretrained_weight)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        eval_acc = 0.0
        eval_loss = 0.0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x.cuda())
            loss = criterion(y_pred, y.cuda())
            loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(y_pred.cpu().data.numpy(), axis=1) == y.numpy())
            train_loss += loss.item()
            print(f'[{i:03d}/{num_train_batch}]', end='\r')
        
        model.eval()
        with torch.no_grad():
            for i, (x,y) in enumerate(eval_loader):
                y_pred = model(x.cuda())
                loss = criterion(y_pred, y.cuda())
                eval_acc += np.sum(np.argmax(y_pred.cpu().data.numpy(), axis=1) == y.numpy())
                eval_loss += loss.item()
                print(f'[{i:03d}/{num_eval_batch}]', end='\r')
        train_acc /= train_size
        train_loss /= train_size
        eval_acc /= eval_size
        eval_loss /= eval_size
        print(f'epoch [{epoch+1:02d}/{args.num_train_epochs}]: {time.time()-epoch_start_time:.2f} sec(s)')
        print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        print(f' eval loss: {eval_loss:.4f},  eval acc: {eval_acc:.4f}')
        
        torch.save(model, args.model_save_path)
    
    return

if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger('')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main(args, logger)