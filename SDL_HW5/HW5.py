#%%
from pymagnitude import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
file_path = "GoogleNews-vectors-negative300.magnitude"
vectors = Magnitude(file_path)
# %%
vectors.dim
# %%
vectors.most_similar("picnic", topn=5+1)

#%%
w_list = list(list(zip(*(vectors.most_similar("picnic", topn=30))))[0])
X = vectors.query(w_list) 
w_df = pd.DataFrame(X)
w_df.index = w_list
w_corr=w_df.corr()
e_values,e_vectors=np.linalg.eig(w_corr)
#Sorting the eigen vectors coresponding to eigen values in descending order
args = (-e_values).argsort()
e_values = e_vectors[args]
e_vectors = e_vectors[:, args]
#Taking first 2 components which explain maximum variance for projecting
new_e_vectors=e_vectors[:,:2]
#Projecting it onto new dimesion with 2 axis
neww_X=np.dot(X,new_e_vectors)

plt.figure(figsize=(13,7))
plt.scatter(neww_X[:6,0],neww_X[:6,1],linewidths=10,color='blue')
plt.scatter(neww_X[6:,0],neww_X[6:,1],linewidths=10,color='red')
plt.xlabel("PC1",size=15)
plt.ylabel("PC2",size=15)
plt.title("Word Embedding Space on Fist 2 PCs",size=20)
vocab=list(w_list)
for i, word in enumerate(vocab):
  plt.annotate(word,xy=(neww_X[i,0],neww_X[i,1]))

#%%
w_list = ['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette'] 
vectors.doesnt_match(w_list)
#%%
X = vectors.query(w_list) 
w_df = pd.DataFrame(X)
w_df.index = w_list
w_corr=w_df.corr()
e_values,e_vectors=np.linalg.eig(w_corr)
#Sorting the eigen vectors coresponding to eigen values in descending order
args = (-e_values).argsort()
e_values = e_vectors[args]
e_vectors = e_vectors[:, args]
#Taking first 2 components which explain maximum variance for projecting
new_e_vectors=e_vectors[:,:2]
#Projecting it onto new dimesion with 2 axis
neww_X=np.dot(X,new_e_vectors)

plt.figure(figsize=(13,7))
plt.scatter(neww_X[:,0],neww_X[:,1],linewidths=10,color='blue')
plt.xlabel("PC1",size=15)
plt.ylabel("PC2",size=15)
plt.title("Word Embedding Space on Fist 2 PCs",size=20)
vocab=list(w_list)
for i, word in enumerate(vocab):
  plt.annotate(word,xy=(neww_X[i,0],neww_X[i,1]))
# %%
X = vectors.most_similar(positive = ["throw", "leg"], negative = ["jump"])
X
# %%
w_list = ["throw", "leg","jump"]
w_list.extend(list(list(zip(*X))[0]))
print(w_list)
#%%
X = vectors.query(w_list) 
w_df = pd.DataFrame(X)
w_df.index = w_list
w_corr=w_df.corr()
e_values,e_vectors=np.linalg.eig(w_corr)
#Sorting the eigen vectors coresponding to eigen values in descending order
args = (-e_values).argsort()
e_values = e_vectors[args]
e_vectors = e_vectors[:, args]
#Taking first 2 components which explain maximum variance for projecting
new_e_vectors=e_vectors[:,:2]
#Projecting it onto new dimesion with 2 axis
neww_X=np.dot(X,new_e_vectors)

plt.figure(figsize=(13,7))
plt.scatter(neww_X[:2,0],neww_X[:2,1],linewidths=10,color='blue')
plt.scatter(neww_X[2:,0],neww_X[2:,1],linewidths=10,color='red')
plt.xlabel("PC1",size=15)
plt.ylabel("PC2",size=15)
plt.title("Word Embedding Space on Fist 2 PCs",size=20)
vocab=list(w_list)
for i, word in enumerate(vocab):
  plt.annotate(word,xy=(neww_X[i,0],neww_X[i,1]))
# %%
