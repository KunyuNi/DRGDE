import torch
import numpy as np
import gc
import time
import pandas as pd
import os
import torch.nn as nn




# Adj: adjacency matrix
# size: the number of required features
# largest: Ture (default) for k-largest (smoothed) and Flase for k-smallest (rough) eigenvalues 选择最大 和最小的
# niter: maximum number of iterations
def cal_spectral_feature(Adj, size,type='drug', largest=False, niter=5,dataset='C',num=0):
	# params for the function lobpcg
	# k: the number of required features
	# largest: Ture (default) for k-largest (smoothed)  and Flase for k-smallest (rough) eigenvalues
	# niter: maximum number of iterations
	# for more information, see https://pytorch.org/docs/stable/generated/torch.lobpcg.html

	value,vector=torch.lobpcg(Adj,k=size, largest=largest,niter=niter)


	if largest==True:
		feature_file_name=dataset+r'/'+str(num)+'_smooth_'+type+'_features.npy'
		value_file_name=dataset+r'/'+str(num)+'_smooth_'+type+'_values.npy'

	else:
		feature_file_name=dataset+r'/'+str(num)+'_rough_'+type+'_features.npy'
		value_file_name=dataset+r'/'+str(num)+'_rough_'+type+'_values.npy'


	np.save(value_file_name,value.cpu().numpy())
	np.save(feature_file_name,vector.cpu().numpy())

def pre(smooth_ratio,rough_ratio,dataset,num):
    path='./datasets/'+dataset # 文件路径
    # 读取文件
    test_data=pd.read_csv(path+'/test_'+str(num)+'.csv')
    rate_matrix=torch.Tensor(np.load(path+'/train_'+str(num) +'.npy')).cuda()
    """ C drug:658, disease:409 association:2520
        F drug:593, disease:313 association:1933
        L drug:598, disease:269 association:18416
        LR drug:763,disease:681 association:3051
    """
 
    if dataset=='C':
        drug_size,diease_size=658,409
    elif dataset=='F':
        drug_size,diease_size=593,313
    elif dataset=='L':
        drug_size,diease_size=598,269
    elif dataset=='LR':
        drug_size,diease_size=763,681
    elif dataset=='GCMM':
        drug_size,diease_size=1519,728
    elif dataset=='DDI':
        drug_size,diease_size=2320,2320
    elif dataset=='DTI':
        drug_size,diease_size=708,1512
    #drug degree and diease degree
    D_u=rate_matrix.sum(1)
    D_i=rate_matrix.sum(0)
    #in the case any users or items have no interactions
    for i in range(drug_size):
        if D_u[i]!=0:
            D_u[i]=1/D_u[i].sqrt()

    for i in range(diease_size):
        if D_i[i]!=0:
            D_i[i]=1/D_i[i].sqrt()
    rate_matrix=D_u.unsqueeze(1)*rate_matrix*D_i
    #clear GPU
    del D_u, D_i 
    gc.collect()
    torch.cuda.empty_cache()

    #drug-drug matrix
    L_u=rate_matrix.mm(rate_matrix.t())
    # smoothed feautes for user-user relations
    cal_spectral_feature(L_u,int(smooth_ratio*drug_size),type='drug', largest=True,dataset=path,num=num)
    #rough feautes for user-user relations
    if rough_ratio!=0:
        cal_spectral_feature(L_u,int(rough_ratio*drug_size),type='drug',largest=False,dataset=path,num=num)
    
    #clear GPU
    del L_u 
    gc.collect()
    torch.cuda.empty_cache()

    #item-item matrix
    L_i=rate_matrix.t().mm(rate_matrix)
    #smoothed feautes for item-item relations
    cal_spectral_feature(L_i,int(smooth_ratio*diease_size),type='diease', largest=True,dataset=path,num=num)
    #rough feautes for item-item relations
    if rough_ratio!=0:
        cal_spectral_feature(L_i,int(rough_ratio*diease_size),type='diease',largest=False,dataset=path,num=num)


