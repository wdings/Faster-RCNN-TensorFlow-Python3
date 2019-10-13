import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnext101_32x8d
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

from sotabencheval.image_classification import ImageNetEvaluator
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = './.data/vision/imagenet'
else: # local settings
    DATA_ROOT = '/home/ubuntu/my_data/'

model = resnext101_32x8d(pretrained=True)

input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ImageNet(
    DATA_ROOT,
    split="val",
    transform=input_transform,
    target_transform=None,
    download=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = model.cuda()
model.eval()

evaluator = ImageNetEvaluator(
                 model_name='Faster-RCNN-TensorFlow-Python3',
                 paper_arxiv_id='1506.01497')

def get_img_id(image_name):
    return image_name.split('/')[-1].replace('.JPEG', '')

with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        input = input.to(device='cuda', non_blocking=True)
        target = target.to(device='cuda', non_blocking=True)
        output = model(input
        image_ids = [get_img_id(img[0]) for img in test_loader.dataset.imgs[i*test_loader.batch_size:(i+1)*test_loader.batch_size]]
        evaluator.add(dict(zip(image_ids, list(output.cpu().numpy()))))

evaluator.save()
