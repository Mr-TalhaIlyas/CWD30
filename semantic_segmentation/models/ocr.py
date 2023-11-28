#%%
import torch, timm, os
from torch import nn
from torch.nn import functional as F
from models.losses import FocalLoss
from models.bricks import resize

def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


def conv1d(in_channel, out_channel):
    layers = [
        nn.Conv1d(in_channel, out_channel, 1, bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


class ResNet_OCR(nn.Module):
    def __init__(self, n_class, feat_channels=[1024, 2048], pretrained='None'):
        super().__init__()

        self.pretrained = pretrained
        if os.path.isfile(self.pretrained):
            pretrained_weights = torch.load(self.pretrained)
            del pretrained_weights['fc.weight']
            del pretrained_weights['fc.bias']
            self.backbone = timm.create_model('resnet101.a1_in1k', pretrained=False,
                                                features_only=True)
            self.backbone.load_state_dict(pretrained_weights, strict=False)
            print('Loaded pretrained weights from {}'.format(self.pretrained))

        elif self.pretrained == 'imagenet':
            self.backbone = timm.create_model('resnet101.a1_in1k', pretrained=True,
                                                features_only=True)
        else:
            self.backbone = timm.create_model('resnet101.a1_in1k', pretrained=False,
                                                features_only=True)
        ch16, ch32 = feat_channels

        self.L = nn.Conv2d(ch16, n_class, 1)
        self.X = conv2d(ch32, 512, 3)

        self.phi = conv1d(512, 256)
        self.psi = conv1d(512, 256)
        self.delta = conv1d(512, 256)
        self.rho = conv1d(256, 512)
        self.g = conv2d(512 + 512, 512, 1)

        self.out = nn.Conv2d(512, n_class, 1)

        self.criterion = FocalLoss()
        self.aux_criterion = FocalLoss()#nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input, target=None):
        input_size = input.shape[2:]
        stg16, stg32 = self.backbone(input)[-2:] #[1, 1024, 32, 32], [1, 2048, 16, 16]
        # resize output feature to make output stride = 8, by default its 16
        stg32 = resize(stg32, scale_factor=4)
        stg16 = resize(stg16, scale_factor=2)
        # print(stg16.shape, stg32.shape)
        X = self.X(stg32) # [1, 512, 16, 16]
        L = self.L(stg16) # [1, K, 32, 32]
        batch, n_class, height, width = L.shape # [1, K, 32, 32]
        l_flat = L.view(batch, n_class, -1) # [1, K, 32*32]
        # M: NKL
        M = torch.softmax(l_flat, -1) # [1, K, 32*32]
        channel = X.shape[1] # 512
        # print(channel)
        X_flat = X.view(batch, channel, -1) # [1, 512, 16*16]
        # print(X_flat.shape, M.shape)
        # f_k: NCK
        f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)

        # query: NKD
        query = self.phi(f_k).transpose(1, 2)
        # key: NDL
        key = self.psi(X_flat)
        logit = query @ key
        # attn: NKL
        attn = torch.softmax(logit, 1)

        # delta: NDK
        delta = self.delta(f_k)
        # attn_sum: NDL
        attn_sum = delta @ attn
        # x_obj = NCHW
        X_obj = self.rho(attn_sum).view(batch, -1, height, width)

        concat = torch.cat([X, X_obj], 1)
        X_bar = self.g(concat)
        out = self.out(X_bar)
        # print(out.shape)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        if self.training:
            aux_out = F.interpolate(
                L, size=input_size, mode='bilinear', align_corners=False
            )

            loss = self.criterion(out, target)
            aux_loss = self.aux_criterion(aux_out, target)

            return {'loss': loss, 'aux': aux_loss}, out

        else:
            return {}, out
        
#%%
# import torch
# model = ResNet_OCR(config['num_classes'], feat_channels=[1024, 2048])

# model = model.to('cuda')
# model.train()

# x = torch.randn((1,3,512,512)).to('cuda')
# t = torch.randint(0, 2, (1,512,512)).to('cuda')
# z,y = model.forward(x, t)
#%%