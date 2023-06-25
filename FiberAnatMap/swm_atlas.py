#create cluster-level anatomical information for ORG dataset(training dataset)
import numpy as np
import re
import glob
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  #isdigt()方法字符串是否全为数字，若全是数字，为True，否则为Fasle.Python lower() 方法转换字符串中所有大写字符为小写
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
regions_path = natural_sort(glob.glob('data/ORG/swm/train/2000_swm_all_105regions_*.npy'))
labels = np.load('data/ORG/swm/train/2000_swm_all_105regions_0.npy')[0,1,:]
region_all = np.zeros((2000,105))
for i in range(100): 
    id = regions_path[i].split('/')[-1].split('.')[0].split('_')[-1]
    regions = np.load(regions_path[i])[:,0,:]  
    #add all rows
    region_all +=regions
np.save('data/ORG/swm/train/swm_100_region_atlas.npy',region_all)  
disMat = sch.distance.pdist(region_all,'euclidean')
#hierarchical clustering
Z=sch.linkage(disMat,method='weighted') 
cluster= sch.fcluster(Z, t=99, criterion='distance')  #t=99 8 classes for SWM,  t=47.8 80 classes for DWM
print(np.unique(cluster)) 

# show clustering tree and save plot_dendrogram.png
plt.figure(figsize=(30,30),num=1)
P=sch.dendrogram(Z)
plt.savefig('plot_dendrogram.png')

index_ = []
for i in range(np.unique(cluster).shape[0]+1):
    index_.append(list(np.where(cluster==i)[0]))
index_ = index_[1:]  #delete 0
print(len(index_))

cluster_new = np.zeros(2000,)
x_first = []
for j in range(len(index_)):
    if len(index_[j])!=0:
        x_first.append(index_[j][0])
print(x_first)
x_first_sort = sorted(x_first)
print(x_first_sort)
#reorder the cluster
for k in range(np.unique(cluster).shape[0]):
    cluster_new[np.where(cluster == cluster[x_first_sort[k]])[0]]+=k
cluster_new = cluster_new.astype(int)
print('---',cluster_new.shape)
print(cluster_new)
np.save('data/ORG/swm/train/swm_100_region_atlas_80cluster.npy',cluster_new)
#The 80 classes are grouped into 8 classes according to ORG atlas
cluster_8 = np.zeros((2000,))
index0_1 = np.where((cluster_new>=0)&(cluster_new<=20))[0]
index0_2 = np.where((cluster_new>=22)&(cluster_new<=30))[0]
index0 = np.concatenate((index0_1,index0_2),axis=0)
print(index0.shape)
index1 = np.where((cluster_new==21)|(cluster_new==31)|(cluster_new==35))[0]
print(index1.shape)
index2 = np.where((cluster_new==34))[0]
print(index2.shape)
index3 = np.where((cluster_new==33))[0]
print(index3.shape)
index4 = np.where((cluster_new==36))[0]
print(index4.shape)
index5 = np.where((cluster_new==32))[0]
print(index5.shape)
index6 = np.where((cluster_new>=37)&(cluster_new<=40))[0]
print(index6.shape)
index7 = np.where((cluster_new>=41)&(cluster_new<=79))[0]
print(index7.shape)
cluster_8[index0] = 0
cluster_8[index1] = 1
cluster_8[index2] = 2
cluster_8[index3] = 3
cluster_8[index4] = 4
cluster_8[index5] = 5
cluster_8[index6] = 6
cluster_8[index7] = 7
np.save('data/ORG/swm/train/swm_100_region_atlas_8cluster.npy',cluster_8)