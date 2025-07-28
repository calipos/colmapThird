import torch
import os, pathlib
import numpy as np

def save(ckpt_dir, optimizer, model, global_step, scheduler=None, keep_latest=5, model_name='model'):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    prev_ckpts = list(pathlib.Path(ckpt_dir).glob('%s-*' % model_name))
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_ckpts) > keep_latest-1:
        for f in prev_ckpts[keep_latest-1:]:
            f.unlink()
    model_path = '%s/%s-%09d.pth' % (ckpt_dir, model_name, global_step)
    
    ckpt = {'optimizer_state_dict': optimizer.state_dict()}
    ckpt['model_state_dict'] = model.state_dict()
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, model_path)
    print("saved a checkpoint: %s" % (model_path))

def load(ckpt_dir, model, step=0, model_name='model'):
    model_name = 'pips2-000200000.pth' 
    path = os.path.join(ckpt_dir, model_name)
    if not os.path.exists(path):
        print('...there is no full checkpoint here!')
        print('-- note this function no longer appends "saved_checkpoints/" before the ckpt_dir --')
    else:
        checkpoint = torch.load(path,map_location=torch.device('cpu'),weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
   
    return step
