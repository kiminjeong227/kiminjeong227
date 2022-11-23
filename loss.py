import torch
import torch.nn.functional as F

def match_accuracy(logits, label_index):
    logit_argmax = logits.argmax(dim=-1)
    return (logit_argmax == label_index)

def log_prob(log_cls, label_index):
    return torch.gather(log_cls, -1, label_index.unsqueeze(dim=-1)).squeeze(dim=-1)

def log_prob_v2(log_cls, label_index):
    trg = F.one_hot(label_index, num_classes=log_cls.size()[-1])
    return (trg * log_cls).sum(dim=-1)

