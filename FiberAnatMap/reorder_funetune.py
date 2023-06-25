#对经过One_stage模型后得到的swm和dwm(跑完test和check)，这个程序得到swm_8_cluster_long_之后构建sf_clusters_train_featMatrix_{}.h5 （见supwam_master中的data/AD_MCI/SWM/data/py)
#源文件在tract_test中的newdata_test_cluster.py中

import numpy as np
import re
import glob
import vtk
from functools import reduce
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from matplotlib import cm
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

atlas = np.load('./data/AD_MCI_S01/2wm_100_region_atlas.npy')

#create_new_atlas
nonzero_num = []
atlas_new = np.zeros((atlas.shape))
for i in range(2000):
    nonzero_index = np.nonzero(atlas[i])[0] 
    nonzero_num.append(nonzero_index.shape[0])
    top_k_idx = np.argpartition(atlas[i], -8)[-8:]
    # Sets all elements except the maximum 8 numbers to 0
    mask = np.zeros_like(atlas[i])
    mask[top_k_idx] = 1
    atlas_new[i] = atlas[i] * mask
print(atlas_new.shape)

atlas_cluster = np.load('./data/AD_MCI_S01/swm_100_region_atlas_8cluster.npy')

unique_atlas_cluster = np.unique(atlas_cluster)
print(unique_atlas_cluster)

test_region_path = natural_sort(glob.glob('./data/AD_MCI_S01/105region_swm_*.npy'))
normalize_atlas_index_add_all = None
for j in range(unique_atlas_cluster.shape[0]):
    index = np.where(atlas_cluster ==unique_atlas_cluster[j])[0]
    atlas_index = atlas_new[index]
    atlas_index_add = np.zeros((1,105))
    for x in range(atlas_index.shape[0]):  #For each class, find the normalization of that class
        atlas_index_add+=atlas_index[x]
    normalize_atlas_index_add = atlas_index_add / (np.sum(atlas_index_add))
    if j == 0:
        normalize_atlas_index_add_all = normalize_atlas_index_add
    else:
        normalize_atlas_index_add_all = np.concatenate((normalize_atlas_index_add_all,normalize_atlas_index_add),axis=0)
print(normalize_atlas_index_add_all.shape)  #(8,105)


for i in range(len(test_region_path)): 
    id = test_region_path[i].split('/')[-1].split('.')[0].split('_')[-2]+'_'+ test_region_path[i].split('/')[-1].split('.')[0].split('_')[-1]
    print(id)
    test_region =np.load(test_region_path[i])
    print('test_region',test_region.shape)
    repeats = test_region.shape[0] / 2000 +1
    new_arr = np.repeat(atlas_cluster, repeats)
    new_arr_ = np.random.choice(new_arr,test_region.shape[0] , replace=False)
    arrIndex = np.array(new_arr_).argsort()
    atlas_cluster_long = new_arr_[arrIndex]  #Interpolate into the same dimension as test region.shape[0]
    print(atlas_cluster_long.shape)
    np.save('./data/AD_MCI_S01/swm_8_cluster_long_{}.npy'.format(id),atlas_cluster_long)
    #Find the distance between each row and each row in normalize_atlas_index_add_all, the smallest being the category
    min_dis_index_lst = []
    for m in range(test_region.shape[0]):
        if np.nonzero(test_region[m])[0].shape[0]!=0:
            test_region_cluster_dis = []
            for n in range(8):
                B = test_region[m]/np.sum(test_region[m])
                A = normalize_atlas_index_add_all[n]
                # cosine similarity
                cos_sim = (np.dot(A, B)) / (np.linalg.norm(A) * np.linalg.norm(B)) 
                test_region_cluster_dis.append(cos_sim)
        else:
            test_region_cluster_dis=[0]
        min_dis_index = int(min(test_region_cluster_dis)) + atlas_cluster_long[m]  #correct the sequence
        
        min_dis_index_lst.append(min_dis_index)
    np.save('./data/AD_MCI_S01/swm_8_cluster_long_correction_{}.npy'.format(id),min_dis_index_lst)
