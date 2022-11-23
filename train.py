import torch
from model import Classifier


ch = 32
num_classes = 10
img_ch = 3


model = Classifier(img_ch=img_ch, ch=ch, num_classes=num_classes)
