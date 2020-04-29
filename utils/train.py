#optimizer, train, eval, early_stopping
import os
import pickle
import sys
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F

from config import TRAINED_PATH


def create_balanced_mask(shape, length):
    assert len(shape) == 3

    mask = torch.zeros(shape)

    for _idx0 in range(shape[0]):
        for _idx1 in range(shape[1]):
            # try:
            num = int(length[_idx0] / 2) + 1
            # except Exception:
            #     print(len(length), _idx0, shape)

            _indexes_1st = random.sample(range(int(shape[2] / 2)), num)
            _indexes_2nd = random.sample(range(int(shape[2] / 2)), num)
            for _rand_idx in _indexes_1st:
                mask[_idx0][_idx1][_rand_idx] = 1
            for _rand_idx in _indexes_2nd:
                mask[_idx0][_idx1][-(_rand_idx + 1)] = 1

    return mask.view(shape)

class TransformerOptimizer():
    
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.static_lr = np.power(d_model, -0.5)
#        self.static_lr = 0.1
        self.warmup_steps = warmup_steps
        
        self.num_step = 1
        
    def step(self):
        self._adjust_learning_rate()
        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def _adjust_learning_rate(self):
        lr = self.static_lr * min(np.power(self.num_step, -0.5), 
                                  (self.num_step * np.power(self.warmup_steps, -1.5)))
#        lr = self.static_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            

def print_progress(loss, epoch, batch, batch_size, dataset_size):
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '+' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Batch Loss ({}): {:.4f}'.format(epoch, batch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()
    
            
def train_epoch(model, optimizer, loader, epoch, device, args, mask=False, mask_ratio=0.5):
    
    true = []
    pred = []
    losses = []
    b_s = loader.batch_size
    d_s = len(loader.dataset)
    
    model.train()
    
    for i, batch in enumerate(loader, 1):
        
        X = batch[0].to(device)
        y = batch[1].to(device)
        
        if isinstance(model, nn.DataParallel):
            inc_lex = model.module.include_lex
        else:
            inc_lex = model.include_lex

        if inc_lex:
            z = batch[2].to(device)
        else:
            z = None

        lengths = batch[-1].float()

        try:
            assert len(lengths) == len(X)
        except Exception:
            print(len(lengths), lengths[0])
            
        optimizer.zero_grad()
        sp = [len(X), args['seq_length'], args['z_length']]

        if mask:
            msk = create_balanced_mask(shape=sp,
                                      length=lengths*mask_ratio).to(device)

        else:
            msk = None

        logits, _, _, residual = model(X, z, context_mask=msk)

        loss = F.cross_entropy(logits, y.argmax(axis=1)) + residual
        loss.backward()
        optimizer.step()
        
        print_progress(loss.item(), epoch, i, b_s, d_s)
        
        pred += logits.argmax(axis=1).tolist()
        true += y.argmax(axis=1).tolist()
        losses.append(loss.item())
        
    accuracy = np.sum([1 for i in range(len(pred)) if pred[i] == true[i]]) / len(pred)
    
    return np.mean(losses), accuracy
    
    
def eval_epoch(model, loader, epoch, device):
    
    true = []
    pred = []
    losses = []
    b_s = loader.batch_size
    d_s = len(loader.dataset)
    
    model.eval()
    
    for i, batch in enumerate(loader, 1):
        
        X = batch[0].to(device)
        y = batch[1].to(device)
#        if model.include_lex:
#            z = batch[2].to(device)
#        else:
#            z = None

        if isinstance(model, nn.DataParallel):
            inc_lex = model.module.include_lex
        else:
            inc_lex = model.include_lex

        if inc_lex:
            z = batch[2].to(device)
        else:
            z = None
        
        logits, con_attn, lex_attn, _ = model(X, z)
        
        loss = F.cross_entropy(logits, y.argmax(axis=1))
        
        pred += logits.argmax(axis=1).tolist()
        true += y.argmax(axis=1).tolist()
        losses.append(loss.item())
        
    accuracy = np.sum([1 for i in range(len(pred)) if pred[i] == true[i]]) / len(pred)
    
    return np.mean(losses), accuracy, con_attn, lex_attn


    
    


def load_model(name):
    
    return


class TrainingManager():
    
    
    def __init__(self, tolerance):
        self.best_score = 0
        self.best_test_score = 0
        self.tolerance = tolerance
        self.max_tol = tolerance
        
        
    def checkpoint(self, model, args, score, test_score):
        if score > self.best_score:
            
            self.best_score = score
            self.best_test_score = test_score
            self.tolerance = self.max_tol
            print('TOLERANCE:', self.tolerance,'Improved! saving model...')

            self._save_model(model, args)
            return True
        else:
            self.tolerance -= 1
            print('TOLERANCE:', self.tolerance, 'Not Improved!')
            if self.tolerance <= 0:
                return False
            else:
                return True
    
    @staticmethod
    
    def _save_model(model, args):
        

        torch.save(model.state_dict(), 
                   os.path.join(TRAINED_PATH, 
                                '{}.model'.format(args['experiment_name'])))

        pickle.dump(args, 
                    open(os.path.join(TRAINED_PATH, 
                                '{}.args'.format(args['experiment_name'])), 'wb'),
                    protocol=2)
    
    
    
    
