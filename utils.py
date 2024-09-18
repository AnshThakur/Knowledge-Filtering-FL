import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import pandas as pd
from dataclasses import dataclass, asdict
import json
from sklearn import metrics
@dataclass
class Params:
    num_sites: int = None
    num_rounds: int = None
    inner_epochs: int = None
    batch_size: int = None
    outer_lr: float = None
    weight_decay: float = None
    inner_lr: float = None

PARAMS_FILE = "params.json"

def save_params(run_path, params):
    with open(f'{run_path}/{PARAMS_FILE}', "w") as f:
        json.dump(asdict(params), f, indent=2)

def get_device():
    if torch.cuda.is_available():
       device=torch.device('cuda:0')
    else:
       device=torch.device('cpu')   
    return device

def combine_grads(G):

    '''Given a list of client gradients, combine them for meta gradient'''
    
    nodes = len(G)
    keys = G[0].keys() # Because the keys should be the same for all models
    
    Meta_grad = deepcopy(G[0])

    for k in keys:
        
        if 'cca' in k.lower():
            continue
        for i in range(1, nodes):
            Meta_grad[k] += G[i][k]

        Meta_grad[k] = Meta_grad[k]/nodes    

    return Meta_grad   


def prediction_binary(model,loader,loss_fn,device):
    P=[]
    L=[]
    model.eval()
    val_loss=0
    for i,batch in enumerate(loader):
        data,labels=batch
        data=data.to(torch.float32).to(device)
        labels=labels.to(torch.float32).to(device)
        
        pred=model(data)[:,0]
        loss=loss_fn(pred,labels)
        val_loss=val_loss+loss.item()

        P.append(pred.cpu().detach().numpy())
        L.append(labels.cpu().detach().numpy())
        
    val_loss=val_loss/len(loader)
    P=np.concatenate(P)  
    L=np.concatenate(L)
    auc=roc_auc_score(L,P)


    # apr = metrics.average_precision_score(L,P)
    precision, recall, _ = metrics.precision_recall_curve(L,P)
    apr = metrics.auc(recall,precision)

    return val_loss,auc,apr



def evaluate_models(client_id, Loaders, net, TL, loss_fn, device, df, B,P, model_path, ae=False, ae_fl=None):
    ''' Given site i, and model net, evaluate the model peformance on the site's val set'''
    
    tl1 = TL[client_id]
    val_loss, val_auc, val_apr = prediction_binary(net, Loaders[client_id][1], loss_fn, device)
    
    if val_auc > B[client_id]:
       B[client_id] = val_auc
       torch.save(net, f'./trained_models/{model_path}/node{client_id}') 
       if ae_fl:
           torch.save(ae_fl, f'./trained_models/AE_Unstructured/node{client_id}') 

    df = pd.concat([df, pd.DataFrame({'Train_Loss': [tl1], 'Val_Loss': [val_loss], 'Val_AUC': [val_auc], 'Val_APR': [val_apr]})], ignore_index=True)
    # df = df.append({'Train_Loss': tl1, 'Val_Loss': val_loss, 'Val_AUC': val_auc,'Val_APR': val_apr}, ignore_index=True)

    return df, B[client_id] ,val_auc,val_apr    




