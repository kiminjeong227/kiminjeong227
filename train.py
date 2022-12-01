import torch
from model import Classifier
import torch.optim as optim
from utils import save, load, AverageMeter
from dataset import load_dataset
from loss import log_prob, match_accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse



def train(args):
    train_dataset, valid_dataset, class_to_idx = load_dataset(dataset=args.dataset, data_dir=args.data_dir)

    model = Classifier(img_ch=args.img_ch, 
                    ch=args.ch, 
                    num_classes=args.num_classes).to(args.device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    if args.cont_training:
        checkpoint = load(args.save_dir)
        init_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    else:
        init_epoch = 1
        global_step = 0

    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()

    test_loss_meter = AverageMeter()
    test_accuracy_meter = AverageMeter()

    for epoch in range(init_epoch, args.num_epochs + 1):
        
        # train
        model.train()
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        pbar = tqdm(enumerate(train_loader), leave=False)
        
        log_str = f'epoch {str(epoch).zfill(3)}'
        
        for i, (x, y) in pbar:
            pbar.set_description(log_str + f' | train [{i + 1}/{len(train_loader)}]', refresh=True)
            x = x.to(args.device)
            y = y.to(args.device)
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
        
        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            
            checkpoint = {'epoch': epoch + 1, 
                        'global_step': global_step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}

            save(checkpoint, save_dir=args.save_dir)
        
        # test
        if epoch % args.eval_freq == 0 or epoch == args.num_epochs:
            model.eval()
            test_loss_meter.reset()
            test_accuracy_meter.reset()
            pbar = tqdm(enumerate(test_loader), leave=False)

            with torch.no_grad():
                for i, (x, y) in pbar:
                    pbar.set_description(log_str + f' | test [{i + 1}/{len(test_loader)}]', refresh=True)
                    x = x.to(args.device)
                    y = y.to(args.device)
                    log_cls = model(x)
                    log_p = log_prob(log_cls, y)
                    loss = - log_p.mean()
                    accuracy = match_accuracy(log_cls, y).float().mean()
                    test_loss_meter.update(loss.item(), n=x.size()[0])
                    test_accuracy_meter.update(accuracy.item(), n=x.size()[0])
                    
            log_str = log_str + f' | valid loss: {test_loss_meter.avg: .5f} | valid accuracy: {test_accuracy_meter.avg: .5f}'
        
        
        print(log_str)



args = argparse.Namespace()
args.batch_size = 512
args.learning_rate = 1e-3
args.data_dir = '../dataset'
args.dataset = 'mnist'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.save_dir = './save'
args.cont_training = False
args.save_freq = 1
args.eval_freq = 2
args.num_epochs = 10
args.img_ch = 1
args.ch = 64
args.num_classes = 10
train(args)
