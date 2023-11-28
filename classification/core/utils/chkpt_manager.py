# from configs.config import config
from termcolor import cprint
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

def load_checkpoint(model, pretrained_path=None):
    if pretrained_path is not None:
        chkpt = torch.load(pretrained_path,
                            map_location='cpu')
        try:
            # load pretrained
            try:
                pretrained_dict = chkpt['model_state_dict']
                print("[INFO] Loaded Model checkpoint:")
                for key, value in chkpt.items():
                    if key != 'model_state_dict':
                        if key !='optimizer_state_dict':
                            print(f"{key}={value}", end='  ')
                print()
            except KeyError:
                pretrained_dict = chkpt
            # load model state dict
            state = model.state_dict()
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
            model.load_state_dict(state)
            cprint('[INFO] Pre-trained state loaded successfully...', 'blue')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        except:
            cprint(f'ERROR in pretrained_dict @ {pretrained_path}', 'red')
    else:
        print('Enter pretrained_dict path.')
    return matched, unmatched

# matched, unmatched = load_pretrained_chkpt(model, pretrained_path=ckpt)
# print(unmatched)

def save_chkpt(config, model, optimizer, epoch=0, loss=0, iou=0, module='model', return_chkpt=False):
    cprint('-> Saved checkpoint', 'green')
    torch.save({
                'epoch': epoch,
                'loss': loss,
                'acc/iou': iou,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(config["checkpoint_path"], f'{module}_{config["experiment_name"]}.pth'))
    cprint(os.path.join(config["checkpoint_path"], f'{module}_{config["experiment_name"]}.pth'), 'cyan')
    
    if return_chkpt:
        return os.path.join(config["checkpoint_path"], f'{module}_{config["experiment_name"]}.pth')

    
def get_summary(model, input_size=(3,224,224), depth=3):
    summary(model, input_size, depth=depth)

