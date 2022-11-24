import torch.nn as nn


class ImgNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2. * (x - 0.5)
    

class Classifier(nn.Module):
    def __init__(self, img_ch=1, ch=64, num_classes=10):
        super().__init__()
        self.img_ch = img_ch 
        self.ch = ch
        self.num_classes = num_classes
        self.conv = nn.Sequential(ImgNorm(),
                                  nn.Conv2d(img_ch, ch, 3, 1, 1),
                                  nn.AvgPool2d(kernel_size=2),
                                  nn.ReLU(),
                                  nn.Conv2d(ch, 2 * ch, 3, 1, 1),
                                  nn.AvgPool2d(kernel_size=2),
                                  nn.ReLU(),
                                  nn.Conv2d(2 * ch, 4 * ch, 3, 1, 1),
                                  nn.AvgPool2d(kernel_size=2),
                                  nn.ReLU(),
                                  nn.Conv2d(4 * ch, 8 * ch, 3, 1, 1),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.ReLU())
        
        self.linear = nn.Sequential(nn.Linear(8 * ch, 4 * ch),
                                    nn.ReLU(),
                                    nn.Linear(4 * ch, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1, -1)
        logits = self.linear(x)
        log_class = logits.log_softmax(dim=-1)
        return log_class
    