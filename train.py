import torch
from model import Classifier
import torch.optim as optim
from utils import save, load, AverageMeter
from dataset import load_dataset
from loss import log_prob, match_accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm



batch_size = 256
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

save_freq = 1
eval_freq = 1
num_epochs = 50


save_dir = './save'
cont_training = False

if cont_training:
    state_dict = load(save_dir)
    init_epoch = state_dict['epoch']
    global_step = state_dict['global_step']
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])

else:
    init_epoch = 1
    global_step = 0



for epoch in range(init_epoch, num_epochs + 1):
    
    # train
    model.train()
    train_loss_meter.reset()
    train_accuracy_meter.reset()
    pbar = tqdm(enumerate(train_loader), leave=False)
    
    log_str = f'epoch {str(epoch).zfill(3)}'
    
    for i, (x, y) in pbar:
        pbar.set_description(log_str + f' | train [{i + 1}/{len(train_loader)}]')
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        log_cls = model(x)
        log_p = log_prob(log_cls, y)
        loss = - log_p.mean()
        loss.backward()
        optimizer.step()
        global_step += 1
        accuracy = match_accuracy(log_cls, y).float().mean()
        train_loss_meter.update(loss.item(), n=x.size()[0])
        train_accuracy_meter.update(accuracy.item(), n=x.size()[0])

    log_str = log_str + f' | train loss: {train_loss_meter.avg: .5f} | train accuracy: {train_accuracy_meter.avg: .5f}'
    
    if epoch % save_freq == 0 or epoch == num_epochs:
        
        state_dict = {'epoch': epoch + 1, 
                      'global_step': global_step,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}

        save(state_dict, save_dir='./save')
    
    # test
    if epoch % eval_freq == 0 or epoch == num_epochs:
        model.eval()
        test_loss_meter.reset()
        test_accuracy_meter.reset()
        pbar = tqdm(enumerate(test_loader), leave=False)

        with torch.no_grad():
            for i, (x, y) in pbar:
                pbar.set_description(log_str + f' | test [{i + 1}/{len(test_loader)}]')
                x = x.to(device)
                y = y.to(device)
                log_cls = model(x)
                log_p = log_prob(log_cls, y)
                loss = - log_p.mean()
                accuracy = match_accuracy(log_cls, y).float().mean()
                test_loss_meter.update(loss.item(), n=x.size()[0])
                test_accuracy_meter.update(accuracy.item(), n=x.size()[0])
                
        log_str = log_str + f' | valid loss: {test_loss_meter.avg: .5f} | valid accuracy: {test_accuracy_meter.avg: .5f}'
    
    
    print(log_str)
