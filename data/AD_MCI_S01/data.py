import h5py
import numpy as np
import re
import glob
import vtk
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  #isdigt()方法字符串是否全为数字，若全是数字，为True，否则为Fasle.Python lower() 方法转换字符串中所有大写字符为小写
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
point_path = natural_sort(glob.glob('*/sub_tract.npy'))

region = np.load('swm_100_ad_atlas_8cluster.npy')

for i in range(len(point_path)):
    point = np.load(point_path[i])
    id = point_path[i].split('/')[-2]
    ind_region = np.load('{}_105regions.npy'.format(id))[:,0,:]
    print(ind_region.shape)  #(10000,105)
    path2 = 'downsam_point_label_region_sort_{}.h5'.format(id)
    h5f = h5py.File(path2, 'w')
    h5f.create_dataset('point', data=point)
    h5f.create_dataset('region',data=region)
    h5f.create_dataset('ind_region',data=ind_region)

h5_path = natural_sort(glob.glob('downsam_point_label_region_sort_*.h5'))
points_20 = None
labels_20 = None
regions_20 = None
ind_regions_20 = None
j_range = list(range(0,125,25))
for j in range(len(j_range)):
    for i in range(25):
        h5_ = h5py.File(h5_path[i],'r')
        points = np.array(h5_['point'])
        regions = np.array(h5_['region'])
        ind_regions = np.array(h5_['ind_region'])
        if i == 0:
            points_20 = points
            regions_20 = regions
            ind_regions_20 = ind_regions
        else:
            points_20 = np.concatenate((points_20,points),axis=0)
            regions_20 = np.concatenate((regions_20,regions),axis=0)
            ind_regions_20 = np.concatenate((ind_regions_20,ind_regions),axis=0)
    path2 = 'sf_clusters_train_featMatrix_{}.h5'.format(j+1)
    h5f = h5py.File(path2, 'w')
    h5f.create_dataset('point', data=points_20)
    h5f.create_dataset('region',data=regions_20)
    h5f.create_dataset('ind_region',data=ind_regions_20)

    
