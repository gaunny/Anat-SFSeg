import numpy as np
import torch
import re
import glob
import torch.utils.data as data
import h5py
import os
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  #isdigt()方法字符串是否全为数字，若全是数字，为True，否则为Fasle.Python lower() 方法转换字符串中所有大写字符为小写
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
#ORGdataset for training
class ORGDataset(data.Dataset):
    def __init__(self, logger, num_fold=1, k=5, split='train'):
        root='data/ORG/swm/train/'
        self.split = split
        self.num_fold = num_fold
        self.k = k
        self.logger = logger
        features_combine = None
        labels_combine = None
        regions_combine = None
        ind_regions_combine = None
        if self.split == 'train':
            train_fold = 0
            train_fold_lst = []
            for i in range(5):
                if i+1 != self.num_fold:
                    feat_h5 = h5py.File(os.path.join(root ,'sf_clusters_train_featMatrix_{}.h5'.format(str(i+1))), 'r')
                    features = np.array(feat_h5['point'])  #(2000,15,3)
                    labels = np.array(feat_h5['label'])   #(2000,)
                    regions = np.array(feat_h5['region'])  #cluster-level anatomical features  (2000,)
                    ind_regions = np.array(feat_h5['ind_region']) #individual-level anatomical features (2000,105)
                    #labels = labels-198  #for dwm
                    if train_fold == 0:
                        features_combine = features
                        labels_combine = labels
                        regions_combine = regions
                        ind_regions_combine = ind_regions
                    else:
                        features_combine = np.concatenate((features_combine, features), axis=0)
                        labels_combine = np.concatenate((labels_combine, labels), axis=0)
                        regions_combine = np.concatenate((regions_combine, regions), axis=0)
                        ind_regions_combine = np.concatenate((ind_regions_combine,ind_regions),axis=0)
                    train_fold_lst.append(i+1)
                    train_fold += 1
            self.features = features_combine
            self.labels = labels_combine
            self.regions = regions_combine
            self.ind_regions = ind_regions_combine
            logger.info('use {} fold as train data'.format(train_fold_lst))
            logger.info('The size of feature for {} is {}'.format(self.split, self.features.shape))
        else:
            feat_h5 = h5py.File(os.path.join(root,'sf_clusters_train_featMatrix_{}.h5'.format(self.num_fold)), 'r')
            self.features = np.array(feat_h5['point'])
            self.labels = np.array(feat_h5['label'])
            self.regions = np.array(feat_h5['region'])
            self.ind_regions = np.array(feat_h5['ind_region'])
            #self.labels = self.labels -198 #for dwm
            logger.info('use {} fold as validation data'.format(self.num_fold))
            logger.info('The size of feature for {} is {}'.format(self.split, self.features.shape))
        self.label_names = [*feat_h5['label_name']]

    def __getitem__(self, index):
        point_set = self.features[index]
        label = self.labels[index]
        region = self.regions[index]
        ind_region = self.ind_regions[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            print('Feature is not in float32 format')
        if label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            print('Label is not in int64 format')
        if region.dtype == 'float32':
            region = torch.from_numpy(np.array([region]))
        else:
            region = torch.from_numpy(np.array([region]).astype(np.float32))
        if ind_region.dtype == 'float32':
            ind_region = torch.from_numpy(np.array([ind_region]))
        else:
            ind_region = torch.from_numpy(np.array([ind_region]).astype(np.float32))
            print('ind_region is not in float32 format')
        return point_set, label,region,ind_region

    def __len__(self):
        return len(self.labels)

    def obtain_label_names(self):
        return self.label_names

#classify SWM and DWM on whole brain tractography
class TestAllDataset(data.Dataset):
    def __init__(self,num):  
        paths = natural_sort(glob.glob('data/AD_MCI_S01/*/sub_tract.npy'))
        features_combine = []
        for i in range(len(paths)):
            id = paths[i].split('/')[-2]
            print(id)
            points = np.load('/data/AD_MCI_S01/{}/sub_tract.npy'.format(id))  #(1000,15,3)
            features_combine.append(points)
        self.points = features_combine[num]

    def __getitem__(self,index):
        point_set = self.points[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
        return point_set
    def __len__(self):
        return self.points.shape[0]
#filter outliers in SWM  
class TestSWMDataset(data.Dataset):
    def __init__(self,num): 
        paths = natural_sort(glob.glob('/data/AD_MCI_S01/SWM/sf_clusters_train_featMatrix_*.h5'))
        features_combine = []
        for i in range(len(paths)):
            id = paths[i].split('/')[-1].split('.')[0].split('_')[-2]+'_'+paths[i].split('/')[-1].split('.')[0].split('_')[-1]
            print(id)
            swm_h5 = h5py.File('/data/AD_MCI_S01/SWM/sf_clusters_train_featMatrix_{}.h5'.format(id),'r')
            points = np.array(swm_h5['point'])
            print(points.shape)
            features_combine.append(points)
        self.points = features_combine[num]

    def __getitem__(self,index):
        point_set = self.points[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))  
        return point_set
    def __len__(self):
        return self.points.shape[0]
#DWM test dataset
class TestDWMDataset(data.Dataset):
    def __init__(self,num): 
        paths = natural_sort(glob.glob('/data/AD_MCI_S01/DWM/sf_clusters_train_featMatrix_*.h5'))
        features_combine = []
        region_combine = []
        ind_region_combine = []
        for i in range(len(paths)):
            id = paths[i].split('/')[-1].split('.')[0].split('_')[-2]+'_'+paths[i].split('/')[-1].split('.')[0].split('_')[-1]
            print(id)
            feat_h5 = h5py.File('/data/AD_MCI_S01/DWM/sf_clusters_train_featMatrix_{}.h5'.format(id), 'r')
            points = np.array(feat_h5['point'])
            regions = np.array(feat_h5['region'])
            ind_regions = np.array(feat_h5['ind_region'])
            if i == 0:
                features_combine = points
                region_combine = regions
            else:
                features_combine = np.concatenate((features_combine,points), axis=0)

            features_combine.append(points)
            region_combine.append(regions)
            ind_region_combine.append(ind_regions)
        self.points = features_combine[num]  
        self.regions = region_combine[num]
        self.ind_regions = ind_region_combine[num]
        
    def __getitem__(self,index):
        point_set = self.points[index]
        region = self.regions[index]
        ind_region = self.ind_regions[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
        if region.dtype == 'float32':
            region = torch.from_numpy(np.array([region]))
        else:
            region = torch.from_numpy(np.array([region]).astype(np.float32))
        if ind_region.dtype == 'float32':
            ind_region = torch.from_numpy(np.array([ind_region]))
        else:
            ind_region = torch.from_numpy(np.array([ind_region]).astype(np.float32))
        return point_set,region,ind_region
    def __len__(self):
        return self.points.shape[0]