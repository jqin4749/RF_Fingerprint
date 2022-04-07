from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class residual_block(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(residual_block, self).__init__()
        self.in_channels_ = in_channels
        self.out_channels_ = out_channels
        self.padding_ = 0
        self.net_ = nn.Sequential(
            nn.Conv2d(self.in_channels_,self.out_channels_,(1,1),padding=0),
            nn.Conv2d(self.out_channels_,self.out_channels_,(3,2),padding=1,dilation=(1,2)),
            nn.BatchNorm2d(self.out_channels_),
            nn.ReLU(),
            nn.Conv2d(self.out_channels_,self.out_channels_,(3,2),padding=1,dilation=(1,2)),
            nn.BatchNorm2d(self.out_channels_)
        )
    def forward(self, x):
        out = self.net_(x)
        x = F.pad(x,(0,0,0,0,0,out.shape[1]-x.shape[1]),mode='replicate')
        out += x
        return F.relu(out)

class feature_extractor(nn.Module):
    def __init__(self) -> None:
        super(feature_extractor, self).__init__()
        self.input_shape_ = (256, 2)
        self.net_ = nn.Sequential(
            residual_block(self.input_shape_[1],16),
            nn.MaxPool2d((2,1)),
            residual_block(16,32),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(32,16,(1,1))
        )
    def forward(self, x):
        out = self.net_(x)
        return out

class feature_proc(nn.Module):
    def __init__(self) -> None:
        super(feature_proc, self).__init__()
        self.net_ = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16, 80),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        return self.net_(x)

class decision_block(nn.Module):
    def __init__(self) -> None:
        super(decision_block, self).__init__()
        self.net_ = nn.Sequential(
            nn.Linear(80, 1),
            nn.Sigmoid() # remove this if use crossEntropyLoss
        )
    def forward(self, x):
        return self.net_(x)


class ova(nn.Module):

    def __init__(self, output_shape) -> None:
        super(ova, self).__init__()
        self.dev_ =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.input_shape_ = (256, 2)
        self.net_ = nn.Sequential(
            feature_extractor(),
            feature_proc()
        )
        self.dbs_ = nn.ModuleList([decision_block() for _ in range(output_shape)])

    def forward(self, class_idx, x): # x: bs, 256, 2
        if x.ndim == 2: # 256, 2
            x = torch.unsqueeze(x,0) # 1, 256, 2
        x = torch.unsqueeze(x, 1) #  add a dimension bs, 1, 256, 2
        x = torch.swapaxes(x,1,3) # put '2' at the channel position. bs, 2, 256, 1
        out = self.net_(x)
        # out = [db(out) for db in self.dbs_]
        out = self.dbs_[class_idx](out)
        return out
    

class encoder(nn.Module):
    def __init__(self) -> None:
        super(encoder, self).__init__()
        self.net_ = nn.Sequential(
            residual_block(1,32),
            nn.MaxPool2d((2,2)),
            residual_block(32,32),
            nn.MaxPool2d((2,1)),
            residual_block(32,32),
            nn.MaxPool2d((2,1)),
            residual_block(32,32),
            residual_block(32,16),
            nn.Conv2d(16,1,(1,1))
        )
    def forward(self, x):
        return self.net_(x)

class decoder(nn.Module):
    def __init__(self) -> None:
        super(decoder, self).__init__()
        self.net_ = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            residual_block(1,16),
            nn.Upsample(scale_factor=(2,1)),
            residual_block(16,16),
            nn.Upsample(scale_factor=(2,1)),
            residual_block(16,16),
            residual_block(16,32),
            residual_block(32,64),
            nn.Conv2d(64,1,(1,1))
        )
    
    def forward(self, x):
        return self.net_(x)

class autoencoder(nn.Module):
    def __init__(self) -> None:
        super(autoencoder, self).__init__()
        self.encoder_ = encoder()
        self.decoder_ = decoder()

    def forward(self, x):
        if x.ndim == 2: # 256, 2
            x = torch.unsqueeze(x,0) # 1, 256, 2
        x = torch.unsqueeze(x, 1)
        features = self.encoder_(x)
        out = self.decoder_(features)
        out = torch.squeeze(out, 1)
        return out