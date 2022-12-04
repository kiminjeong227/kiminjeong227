import torch
from model import Classifier
import torch.optim as optim
from utils import save, load, AverageMeter
from dataset import load_dataset
from loss import log_prob, match_accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import logging


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
    
    

    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard'))
    
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    log_format = '%(asctime)s - %(message)s'
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    test_loss_meter = AverageMeter()
    test_accuracy_meter = AverageMeter()
    
    
    logger.info(f'epoch {init_epoch} - step {global_step} - start training')

    for epoch in range(init_epoch, args.num_epochs + 1):
        
        # train
        model.train()
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        pbar = tqdm(train_loader, leave=False)
        
        log_str = f'epoch {str(epoch).zfill(3)}'
        
        for i, (x, y) in enumerate(pbar):
            pbar.set_description(log_str + f' | train ', refresh=True)
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            log_cls = model(x)
            log_p = log_prob(log_cls, y)
            loss = - log_p.mean()
            loss.backward()
            optimizer.step()
            accuracy = match_accuracy(log_cls, y).float().mean()
            train_loss_meter.update(loss.item(), n=x.size()[0])
            train_accuracy_meter.update(accuracy.item(), n=x.size()[0])
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/accuracy', accuracy.item(), global_step)
            logger.info(f'epoch {epoch} - step {global_step} - train loss {loss.item(): .5f}')
            logger.info(f'epoch {epoch} - step {global_step} - train accuracy {accuracy.item(): .5f}')
            
            global_step += 1

        pbar.close()
        log_str = log_str + f' | train loss: {train_loss_meter.avg: .5f} | train accuracy: {train_accuracy_meter.avg: .5f}'
        
        # save
        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            
            checkpoint = {'args': args,
                          'epoch': epoch + 1, 
                          'global_step': global_step,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}

            save(checkpoint, save_dir=args.save_dir)
            logger.info(f'epoch {epoch} - checkpoint saved')

        # test
        if epoch % args.eval_freq == 0 or epoch == args.num_epochs:
            model.eval()
            test_loss_meter.reset()
            test_accuracy_meter.reset()
            pbar = tqdm(test_loader, leave=False)

            with torch.no_grad():
                for i, (x, y) in enumerate(pbar):
                    pbar.set_description(log_str + f' | test ', refresh=True)
                    x = x.to(args.device)
                    y = y.to(args.device)
                    log_cls = model(x)
                    log_p = log_prob(log_cls, y)
                    loss = - log_p.mean()
                    accuracy = match_accuracy(log_cls, y).float().mean()
                    test_loss_meter.update(loss.item(), n=x.size()[0])
                    test_accuracy_meter.update(accuracy.item(), n=x.size()[0])

            pbar.close()
            writer.add_scalar('valid/loss', test_loss_meter.avg, epoch)
            writer.add_scalar('valid/accuracy', test_accuracy_meter.avg, epoch)
            logger.info(f'epoch {epoch} - valid loss {test_loss_meter.avg: .5f}')
            logger.info(f'epoch {epoch} - valid accuracy {test_accuracy_meter.avg: .5f}')
            
            log_str = log_str + f' | valid loss: {test_loss_meter.avg: .5f} | valid accuracy: {test_accuracy_meter.avg: .5f}'
        
        print(log_str)
        
    writer.close()


args = argparse.Namespace()
args.batch_size = 512
args.learning_rate = 1e-3
args.data_dir = '../dataset'
args.dataset = 'fashion_mnist'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.save_dir = './save'
args.cont_training = False
args.save_freq = 2
args.eval_freq = 1
args.num_epochs = 20
args.img_ch = 1
args.ch = 64
args.num_classes = 10
train(args)
