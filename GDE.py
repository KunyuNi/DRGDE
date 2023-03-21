import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import gc
import pytorch_lightning as pl
from sklearn import metrics
import time 
from preprocess import pre 
from split import split


# specicy the dataset, batch_size, and learning_rate.
batch_size=128
learning_rate=1.3
dataset='C'
smooth_ratio,rough_ratio=0.33,0.02
# C
torch.manual_seed(666)
# 数据部分
if dataset=='C':
	user_size,item_size=658,409
elif dataset=='F':
	user_size,item_size=593,313
elif dataset=='L':
	user_size,item_size=598,269
elif dataset=='LR':
	user_size,item_size=763,681
elif dataset=='GCMM':
    user_size,item_size=1519,728
elif dataset=='DDI':
    user_size,item_size=2320,2320
elif dataset=='DTI':
    user_size,item_size=708,1512
path='./datasets/'+dataset # 文件路径


AUROC,AUPR=[],[]
   
class GDE(nn.Module):
	def __init__(self, user_size, item_size,u=0, beta=1.8, feature_type='both', drop_out=0.1, latent_size=128, reg=0.005):
		super(GDE, self).__init__()
		self.user_embed=torch.nn.Embedding(user_size,latent_size)
		self.item_embed=torch.nn.Embedding(item_size,latent_size)

		nn.init.xavier_normal_(self.user_embed.weight)
		nn.init.xavier_normal_(self.item_embed.weight)

		self.beta=beta
		self.reg=reg  # 正则化
		self.drop_out=drop_out
		if drop_out!=0:
			self.m=torch.nn.Dropout(drop_out)

		if feature_type=='smoothed':
			# 对特征值 操作
			user_filter=self.weight_feature(torch.Tensor(np.load(path+r'/'+str(u)+'_smooth_drug_values.npy')).cuda())
			item_filter=self.weight_feature(torch.Tensor(np.load(path+r'/'+str(u)+'_smooth_diease_values.npy')).cuda())

			user_vector=torch.Tensor(np.load(path+r'/'+str(u)+'_smooth_drug_features.npy')).cuda()
			item_vector=torch.Tensor(np.load(path+r'/'+str(u)+'_smooth_diease_features.npy')).cuda()
		elif feature_type=='rough':
			user_filter=self.weight_feature(torch.Tensor(np.load(path+r'/'+str(u)+'_rough_drug_values.npy')).cuda())
			item_filter=self.weight_feature(torch.Tensor(np.load(path+r'/'+str(u)+'_rough_diease_values.npy')).cuda())

			user_vector=torch.Tensor(np.load(path+r'/'+str(u)+'_rough_drug_features.npy')).cuda()
			item_vector=torch.Tensor(np.load(path+r'/'+str(u)+'_rough_diease_features.npy')).cuda()	

		elif feature_type=='both':
			
			

			user_filter=torch.cat([self.weight_feature(torch.Tensor(np.load(path+r'/'+str(u)+'_smooth_drug_values.npy')).cuda())\
				,self.weight_feature(torch.Tensor(np.load(path+r'/'+str(u)+'_rough_drug_values.npy')).cuda())])

			item_filter=torch.cat([self.weight_feature(torch.Tensor(np.load(path+r'/'+str(u)+'_smooth_diease_values.npy')).cuda())\
				,self.weight_feature(torch.Tensor(np.load(path+r'/'+str(u)+'_rough_diease_values.npy')).cuda())])


			user_vector=torch.cat([torch.Tensor(np.load(path+r'/'+str(u)+'_smooth_drug_features.npy')).cuda(),\
				torch.Tensor(np.load(path+r'/'+str(u)+'_rough_drug_features.npy')).cuda()],1)


			item_vector=torch.cat([torch.Tensor(np.load(path+r'/'+str(u)+'_smooth_diease_features.npy')).cuda(),\
				torch.Tensor(np.load(path+r'/'+str(u)+'_rough_diease_features.npy')).cuda()],1)


		else:
			print('error')
			exit()

		self.L_u=(user_vector*user_filter).mm(user_vector.t()) # beta 特征值* 特征向量 * 特征向量
		self.L_i=(item_vector*item_filter).mm(item_vector.t())


		del user_vector,item_vector,user_filter, item_filter
		gc.collect()
		torch.cuda.empty_cache()

	
	def weight_feature(self,value):
		return torch.exp(self.beta*value)	


	def forward(self, user, pos_item, nega_item, loss_type='no_adaptive'):

		if self.drop_out==0:
			final_user,final_pos,final_nega=self.L_u[user].mm(self.user_embed.weight),self.L_i[pos_item].mm(self.item_embed.weight),self.L_i[nega_item].mm(self.item_embed.weight)

		else:
			final_user,final_pos,final_nega=(self.m(self.L_u[u])*(1-self.drop_out)).mm(self.user_embed.weight),(self.m(self.L_i[p])*(1-self.drop_out)).mm(self.item_embed.weight),\
			(self.m(self.L_i[nega])*(1-self.drop_out)).mm(self.item_embed.weight)


		if loss_type=='adaptive':
			res_nega=(final_user*final_nega).sum(1)
			nega_weight=(1-(1-res_nega.sigmoid().clamp(max=0.99)).log10()).detach()
			out=((final_user*final_pos).sum(1)-nega_weight*res_nega).sigmoid()

		else:	
			out=((final_user*final_pos).sum(1)-(final_user*final_nega).sum(1)).sigmoid()# shape 是 对每个采样
		# 隐藏层 大小 64  batch_size 256
		reg_term=self.reg*(final_user**2+final_pos**2+final_nega**2).sum() 
		return (-torch.log(out).sum()+reg_term)/batch_size

	# 对最终矩阵做一个预测
	def predict_matrix(self):

		final_user=self.L_u.mm(self.user_embed.weight)
		final_item=self.L_i.mm(self.item_embed.weight)
		#mask the observed interactions
		return (final_user.mm(final_item.t())).sigmoid()
	# 改
	def test2(self):
		predict=self.predict_matrix() # predict 
		predict1=torch.clone(predict)
		predict1=predict1.cpu().reshape(-1).detach().numpy()
		label=torch.clone(rate_matrix)
		label=label.cpu().reshape(-1).detach().numpy()

		aupr = metrics.average_precision_score(y_true=label, y_score=predict1) 
		auroc = metrics.roc_auc_score(y_true=label, y_score=predict1)
		AUROC.append(auroc)
		AUPR.append(aupr)
		print("auroc:",auroc,"aupr",aupr)
  
# 预处理 
split(dataset=dataset,n_splits=10,seed=666)


for u in range(10):
	# 加载数据
	print(u,'fold',end=' ')
	df_test=pd.read_csv(path+r'/test_'+str(u)+'.csv')
	rate_matrix=torch.Tensor(np.load(path+r'/train_'+str(u)+'.npy')).cuda()
	train_samples=int(rate_matrix.sum())
	test_data=[[] for i in range(user_size)]
	# print(len(test_data))
	for row in df_test.itertuples():
		test_data[row[1]].append(row[2])
	rate_matrix1=rate_matrix+1e-5
	rate_matrix2=rate_matrix-1e-5
	# 平滑处理
	pre(smooth_ratio,rough_ratio,dataset,u)

	# 准备模型
	model = GDE(user_size, item_size,u).cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
	epoch=train_samples//batch_size
	start_time=time.time()
	for i in range(1000):
		total_loss=0.0
		#start=time.time()
		for j in range(0,epoch):
			# u,p,nega 都是  随机数  
			
			u=torch.LongTensor(np.random.randint(0,user_size,batch_size)).cuda() #0, 最小值，user_size最大值,长度
			p=torch.multinomial(rate_matrix1[u],1,True).squeeze(1)  # 
			nega=torch.multinomial(1-rate_matrix2[u],1,True).squeeze(1)
		
			#  user,item,beta
			loss=model(u,p,nega)
			loss.backward()
			optimizer.step() 
			optimizer.zero_grad()
			total_loss+=loss.item()

		#end=time.time()
		#print(end-start)
		
		print(' epoch %d training loss:%f' %(i,total_loss/epoch))
		
		if (i+1)%20==0 and (i+1)>=160:
			model.test2()
	end_time=time.time()
	print("time:",end_time-start_time)
df = pd.DataFrame({'AUROC': [np.mean(AUROC)], 'AUPR ': [np.mean(AUPR)]})
print('AUROC',np.mean(AUROC), 'AUPR ', np.mean(AUPR))
df.to_csv(dataset+'_end.csv', mode='a', header=False, index=False)



