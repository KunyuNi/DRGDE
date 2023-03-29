
import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import KFold
import h5py


def split(dataset,n_splits,seed):
    if dataset=='C':
        df_train=pd.read_csv("./Cdataset_admat_dgc.txt", sep="\t", index_col=0).T
    elif dataset=='F':
        df_train=pd.read_csv("./Fdataset_admat_dgc.txt", sep="\t", index_col=0).T
    elif dataset=='L':
        df_train=pd.read_csv("./Ldataset.csv",header=None).T
    elif dataset=='LR':
        df_train=pd.read_csv("./LRdataset.txt", sep="\t", index_col=0)
    elif dataset=='GCMM':
        df_train=pd.DataFrame(np.load('./GCMM.npy'))
    elif dataset=='DDI':
        df_train=pd.DataFrame(np.load('./DDI.npy'))
    elif dataset=='DTI':
        df_train=pd.read_csv('./DTI.txt',header=None,sep=' ').T
    elif dataset=='dataset1':
        f = h5py.File("dataset1.h5", "r")
        df_train=pd.DataFrame(f[list(f.keys())[0]])
    elif dataset=='dataset2':
        f = h5py.File("dataset2.h5", "r")
        df_train=pd.DataFrame(f[list(f.keys())[0]])
    elif dataset=='dataset3':
        f = h5py.File("dataset3.h5", "r")
        df_train=pd.DataFrame(f[list(f.keys())[0]])
    elif dataset=='dataset4':
        f = h5py.File("dataset4.h5", "r")
        df_train=pd.DataFrame(f[list(f.keys())[0]])
    elif dataset=='dataset5':
        f = h5py.File("dataset5.h5", "r")
        df_train=pd.DataFrame(f[list(f.keys())[0]])
        
    interaction=np.array(df_train)
    kfold=KFold(n_splits=n_splits,shuffle=True,random_state=seed)
    pos_row, pos_col = np.nonzero(interaction)  # 已知关联的横纵坐标
    neg_row, neg_col = np.nonzero(1 - interaction)  # 未知关联的横纵坐标
    assert len(pos_row) + len(neg_row) == np.prod(interaction.shape)  # 检验正负样本总数是否正确
                # 9:1划分训练集和测试集
    i=0
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row), kfold.split(neg_row)):
        rate_matrix=np.zeros_like(interaction)
        rate_matrix[pos_row[train_pos_idx],pos_col[train_pos_idx]]=1 # 训练集
        # train=np.array(list(zip(pos_row[train_pos_idx],pos_col[train_pos_idx]))) # 测试
        test=np.array(list(zip(pos_row[test_pos_idx],pos_col[test_pos_idx])))
        # 存储
        path='./datasets/'+dataset
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+r'/train_'+str(i)+'.npy',rate_matrix)

        
        # 测试
        # train_pd=pd.DataFrame(train)
        # train_pd.to_csv(path+r'/train_'+str(i)+'.csv',index=False)
        test_pd=pd.DataFrame(test)
        test_pd.to_csv(path+r'/test_'+str(i)+'.csv',index=False)
        i+=1


split('C',10,666)
        
    

    