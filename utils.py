import os
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        

def save(model, ema_model=None, optimizer=None, ckpt_dir='./checkpoint', file_name='model.pth'):
    
    assert os.path.splitext(file_name)[-1] == '.pth', 'invalid file_name'
    
    if not os.path.exists(ckpt_dir):
        print(f'make directory: {ckpt_dir}')
        os.makedirs(ckpt_dir)
    
    state_dict = {'model': model.state_dict()}

    if ema_model is not None:
        state_dict.update({'ema_model': ema_model.state_dict()})

    if optimizer is not None:
        state_dict.update({'optimizer': optimizer.state_dict()})
        
    torch.save(state_dict, f=f'{ckpt_dir}/{file_name}')
    


def load(model, ema_model=None, optimizer=None, ckpt_dir='./checkpoint', file_name='model.pth'):
    
    assert os.path.splitext(file_name)[-1] == '.pth', 'invalid file_name'
    
    state_dict = torch.load(f'{ckpt_dir}/{file_name}')
    model.load_state_dict(state_dict['model'])
    if ema_model is not None:
        ema_model.load_state_dict(state_dict['ema_model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])

    return model, ema_model, optimizer
