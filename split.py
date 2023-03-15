import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from sklearn.model_selection import KFold  


def split(dataset):
    if dataset=='C':
        df_train=pd.read_csv("./Cdataset_admat_dgc.txt", sep="\t", index_col=0).T
    elif dataset=='F':
        df_train=pd.read_csv("./Fdataset_admat_dgc.txt", sep="\t", index_col=0).T
    elif dataset=='L':
        df_train=pd.read_csv("./Ldataset.csv",header=None).T
        
    
    
