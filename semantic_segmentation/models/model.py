#%%
import yaml, math, os
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch, timm
import torch.nn.functional as F
import torch.nn as nn

from models.backbone import MSCANet
from models.decoder import HamDecoder, DecoderHead

pretrained_chkpt = config['encoder_chkpt']

class SegNext(nn.Module):
    '''Different Decoder then SegNext'''
    def __init__(self, num_classes, in_channnels=3, embed_dims=[32, 64, 160, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256, ls_init_val=1e-2, drop_path=0.0, drop_path_mode='row',
                 config=config):
        super().__init__()
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                               ls_init_val=ls_init_val, drop_path=drop_path)
        self.decoder = DecoderHead(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
        
        self.init_weights()
        if pretrained_chkpt is not None:
            self.encoder_init_weights()
        

    def forward(self, x):

        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        output = self.cls_conv(dec_out)

        return output

    def init_weights(self):
        print('Initializing weights...')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)
                # xavier_uniform_() tf default

    def encoder_init_weights(self, pretrained_path=pretrained_chkpt):

        print('Encoder init_weights...')
        chkpt = torch.load(pretrained_path,
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # load pretrained
            pretrained_dict = chkpt['model_state_dict']
            # load model state dict
            state = self.encoder.state_dict()
            # loop over both dicts and make a new dict where name and the shape of new state match
            # with the pretrained state dict.
            matched, unmatched = [], []
            new_dict = {}
            for i, j in zip(pretrained_dict.items(), state.items()):
                pk, pv = i # pretrained state dictionary
                nk, nv = j # new state dictionary
                # if name and weight shape are same
                if pk.strip('module.') == nk.strip('module.') and pv.shape == nv.shape:
                    new_dict[nk] = pv
                    matched.append(pk)
                else:
                    unmatched.append(pk)

            state.update(new_dict)
            self.encoder.load_state_dict(state)
            print('Pre-trained state loaded successfully (Encoder), summary...')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        except:
            print(f'ERROR in pretrained_dict @ {pretrained_path}')
#%%
class SegNext_old(nn.Module):
    def __init__(self, num_classes, in_channnels=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256, config=config, ls_init_val=0.0, drop_path=0.0):
        super().__init__()
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                               ls_init_val=ls_init_val, drop_path=drop_path)
        self.decoder = HamDecoder(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)

    def forward(self, x):

        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        output = self.cls_conv(dec_out)  # here output will be B x C x H/8 x W/8
        output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=True) #now its same as input
        #  bilinear interpol was used originally
        return output
