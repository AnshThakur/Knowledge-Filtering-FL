import torch
from torch.utils.data import TensorDataset, DataLoader
import _pickle as cPickle
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split






def get_loaders(cols,path='./final_data/OUH.csv',id=0,batch=64,sampler=True):
    train=pd.read_csv(path)
    label=train['Covid-19 Positive'].fillna(0)
    train=train[cols]
    print(train.shape)

    T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
    T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)

    T=T.fillna(T.median())
    

    # print(T.median())
    print('---------------------------------------')
    Val_T=Val_T.fillna(T.median())
    Test_T=Test_T.fillna(T.median())


    train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
    train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
    val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
    val_loader = DataLoader(val_dataset, batch_size=batch)  

    test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
    test_loader = DataLoader(test_dataset, batch_size=batch)  

    Loaders=[]
    Loaders.append(train_loader)
    Loaders.append(val_loader)
    Loaders.append(test_loader)
    return Loaders











def get_loaders_structured(cols1,files,path='./final_data/',batch=64,unstruct=0):
    Loaders=[]
    temp=cols1
    for f in files:
        cols=temp
        train=pd.read_csv(path+f+'.csv')
        label=train['Covid-19 Positive'].fillna(0)

        if unstruct==1:
           if (f=='OUH' or f=='UHB'):
              columns1 = [col for col in train.columns if 'Vital_Sign' in col]
              cols=cols+columns1
           
        train=train[cols]      

        print(train.shape)
        print('------------') 
        T, Val_T, L, Val_L = train_test_split(train, label, test_size=0.25, random_state=42)
        T, Test_T, L, Test_L = train_test_split(T, L, test_size=0.25, random_state=42)
        T=T.fillna(T.median())

        Val_T=Val_T.fillna(T.median())
        Test_T=Test_T.fillna(T.median())

        train_dataset= TensorDataset(torch.tensor(T.to_numpy()),torch.tensor(L.to_numpy()))   
        train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=True)  
        
        val_dataset= TensorDataset(torch.tensor(Val_T.to_numpy()),torch.tensor(Val_L.to_numpy()))   
        val_loader = DataLoader(val_dataset, batch_size=batch)  

        test_dataset= TensorDataset(torch.tensor(Test_T.to_numpy()),torch.tensor(Test_L.to_numpy()))   
        test_loader = DataLoader(test_dataset, batch_size=batch)  
      

        Loaders.append([train_loader,val_loader,test_loader]) 

    return Loaders    



