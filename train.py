import torch
from model import Classifier
import torch.optim as optim
from utils import save, load, AverageMeter
from dataset import load_dataset
from loss import log_prob, match_accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm


start_iter = 1
num_epochs = 10
batch_size = 512
learning_rate = 1e-3
data_dir = '../dataset'
device  = 'cuda' if torch.cuda.is_available() else 'cpu'



train_dataset, valid_dataset, class_to_idx = load_dataset(dataset='mnist', data_dir=data_dir)
model = Classifier(img_ch=1, ch=64, num_classes=10).to(device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_meter = AverageMeter()
train_accuracy_meter = AverageMeter()

test_loss_meter = AverageMeter()
test_accuracy_meter = AverageMeter()

for epoch in range(start_iter, start_iter + num_epochs):
    
    # train
    model.train()
    train_loss_meter.reset()
    train_accuracy_meter.reset()
    pbar = tqdm(enumerate(train_loader), leave=False)
    
    for i, (x, y) in pbar:
        pbar.set_description(f'epoch {epoch} train [{i + 1}/{len(train_loader)}]')
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        log_cls = model(x)
        log_p = log_prob(log_cls, y)
        loss = - log_p.mean()
        loss.backward()
        optimizer.step()
        accuracy = match_accuracy(log_cls, y).float().mean()
        train_loss_meter.update(loss.item(), n=x.size()[0])
        train_accuracy_meter.update(accuracy.item(), n=x.size()[0])
        
    
    # test
    model.eval()
    test_loss_meter.reset()
    test_accuracy_meter.reset()
    pbar = tqdm(enumerate(test_loader), leave=False)
    
    with torch.no_grad():
        for i, (x, y) in pbar:
            pbar.set_description(f'epoch {epoch} test [{i + 1}/{len(test_loader)}]')
            x = x.to(device)
            y = y.to(device)
            log_cls = model(x)
            log_p = log_prob(log_cls, y)
            loss = - log_p.mean()
            accuracy = match_accuracy(log_cls, y).float().mean()
            test_loss_meter.update(loss.item(), n=x.size()[0])
            test_accuracy_meter.update(accuracy.item(), n=x.size()[0])
            
    
    print(f'epoch {epoch}'
          + f' | train loss: {train_loss_meter.avg: .5f} | train accuracy: {train_accuracy_meter.avg: .5f}'
          + f' | valid loss: {test_loss_meter.avg: .5f} | valid accuracy: {test_accuracy_meter.avg: .5f}')
