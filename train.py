import torch
from model import Classifier
import torch.optim as optim
from utils import save, load

ch = 32
num_classes = 11
img_ch = 3
learning_rate = 1e-3

model = Classifier(img_ch=img_ch, ch=ch, num_classes=num_classes)
print(next(model.parameters())[0])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

ckpt_dir = './checkpoint'
file_name = 'model.pth'
save(model=model, optimizer=optimizer, ckpt_dir=ckpt_dir, file_name=file_name)

model, _, optimizer = load(model=model, optimizer=optimizer, ckpt_dir=ckpt_dir, file_name=file_name)
