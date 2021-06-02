from torchvision.datasets import ImageFolder

from predict import predict


if __name__ == "__main__":
    dataset = ImageFolder("filtered_imgs/")

    corrects = 0
    for img, label in dataset:
        pred = predict(img)
        print(pred, dataset.classes[label])
        corrects += (pred == dataset.classes[label])
    
    print(f"acc: {round(corrects / len(dataset), 4)}")
