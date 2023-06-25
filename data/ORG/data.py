import h5py
import numpy as np
import re
import glob
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
point_path = natural_sort(glob.glob('2000_swm_all_features_*.npy'))
fiber = ['cluster_00205', 'cluster_00222', 'cluster_00232', 'cluster_00246', 'cluster_00259', 'cluster_00261', 'cluster_00262', 'cluster_00267', 'cluster_00275', 'cluster_00276', 'cluster_00278', 'cluster_00287', 'cluster_00291', 'cluster_00293', 'cluster_00294', 'cluster_00296', 'cluster_00298', 'cluster_00299', 'cluster_00300', 'cluster_00303', 'cluster_00304', 'cluster_00319', 'cluster_00322', 'cluster_00333', 'cluster_00338', 'cluster_00340', 'cluster_00344', 'cluster_00346', 'cluster_00350', 'cluster_00352', 'cluster_00359', 'cluster_00360', 'cluster_00361', 'cluster_00365', 'cluster_00368', 'cluster_00373', 'cluster_00374', 'cluster_00375', 'cluster_00377', 'cluster_00380', 'cluster_00385', 'cluster_00388', 'cluster_00389', 'cluster_00390', 'cluster_00394', 'cluster_00400', 'cluster_00401', 'cluster_00405', 'cluster_00407', 'cluster_00408', 'cluster_00409', 'cluster_00412', 'cluster_00445', 'cluster_00473', 'cluster_00585', 'cluster_00589', 'cluster_00591', 'cluster_00593', 'cluster_00596', 'cluster_00603', 'cluster_00604', 'cluster_00607', 'cluster_00612', 'cluster_00616', 'cluster_00619', 'cluster_00629', 'cluster_00630', 'cluster_00635', 'cluster_00637', 'cluster_00639', 'cluster_00642', 'cluster_00645', 'cluster_00647', 'cluster_00652', 'cluster_00653', 'cluster_00656', 'cluster_00657', 'cluster_00661', 'cluster_00662', 'cluster_00667', 'cluster_00762', 'cluster_00201', 'cluster_00217', 'cluster_00239', 'cluster_00320', 'cluster_00391', 'cluster_00398', 'cluster_00415', 'cluster_00477', 'cluster_00478', 'cluster_00479', 'cluster_00078', 'cluster_00079', 'cluster_00083', 'cluster_00090', 'cluster_00094', 'cluster_00096', 'cluster_00101', 'cluster_00086', 'cluster_00095', 'cluster_00097', 'cluster_00106', 'cluster_00553', 'cluster_00554', 'cluster_00569', 'cluster_00015', 'cluster_00017', 'cluster_00018', 'cluster_00023', 'cluster_00025', 'cluster_00030', 'cluster_00035', 'cluster_00039', 'cluster_00042', 'cluster_00046', 'cluster_00055', 'cluster_00059', 'cluster_00061', 'cluster_00274', 'cluster_00308', 'cluster_00337', 'cluster_00362', 'cluster_00369', 'cluster_00392', 'cluster_00414', 'cluster_00419', 'cluster_00420', 'cluster_00421', 'cluster_00422', 'cluster_00427', 'cluster_00430', 'cluster_00431', 'cluster_00436', 'cluster_00439', 'cluster_00444', 'cluster_00447', 'cluster_00448', 'cluster_00450', 'cluster_00456', 'cluster_00458', 'cluster_00460', 'cluster_00463', 'cluster_00467', 'cluster_00483', 'cluster_00484', 'cluster_00007', 'cluster_00036', 'cluster_00054', 'cluster_00062', 'cluster_00064', 'cluster_00065', 'cluster_00067', 'cluster_00071', 'cluster_00075', 'cluster_00084', 'cluster_00093', 'cluster_00001', 'cluster_00002', 'cluster_00003', 'cluster_00006', 'cluster_00008', 'cluster_00010', 'cluster_00012', 'cluster_00019', 'cluster_00022', 'cluster_00026', 'cluster_00029', 'cluster_00031', 'cluster_00043', 'cluster_00051', 'cluster_00052', 'cluster_00073', 'cluster_00076', 'cluster_00082', 'cluster_00107', 'cluster_00432', 'cluster_00440', 'cluster_00455', 'cluster_00714', 'cluster_00791', 'cluster_00119', 'cluster_00121', 'cluster_00155', 'cluster_00556', 'cluster_00557', 'cluster_00689', 'cluster_00692', 'cluster_00718', 'cluster_00725', 'cluster_00727', 'cluster_00729', 'cluster_00733', 'cluster_00738', 'cluster_00740', 'cluster_00744', 'cluster_00753', 'cluster_00777', 'cluster_00795']
print(len(fiber))
region = np.load('swm_100_region_atlas_8cluster.npy')

for i in range(len(point_path)):
    point = np.load(point_path[i])
    
    id = point_path[i].split('/')[-1].split('.')[0].split('_')[-1]
    label_path ='2000_swm_all_labels_{}.npy'.format(id)
    label = np.load(label_path)
    print(np.unique(label).shape)
    ind_region = np.load('2000_swm_all_105regions_{}.npy'.format(id))
    ind_region = ind_region[:,0,:]
    print(ind_region.shape)  #(2000,105)
    path2 = 'downsam_point_label_region_sort_{}.h5'.format(id)
    h5f = h5py.File(path2, 'w')
    h5f.create_dataset('point', data=point)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('label_name',data=fiber)
    h5f.create_dataset('region',data=region)
    h5f.create_dataset('ind_region',data=ind_region)

h5_path = natural_sort(glob.glob('downsam_point_label_region_sort_*.h5'))

j_range = list(range(0,100,20))
for j in range(len(j_range)):
    points_20 = None
    labels_20 = None
    regions_20 = None
    ind_regions_20 = None
    for i in range(20):

        h5_ = h5py.File(h5_path[i+j_range[j]],'r')
        points = np.array(h5_['point'])
        labels = np.array(h5_['label'])
        regions = np.array(h5_['region'])
        ind_regions = np.array(h5_['ind_region'])
        if i == 0:
            points_20 = points
            labels_20 = labels
            regions_20 = regions
            ind_regions_20 = ind_regions
        else:
            points_20 = np.concatenate((points_20,points),axis=0)
            labels_20 = np.concatenate((labels_20,labels),axis=0)
            regions_20 = np.concatenate((regions_20,regions),axis=0)
            ind_regions_20 = np.concatenate((ind_regions_20,ind_regions),axis=0)
    path2 = 'sf_clusters_train_featMatrix_{}.h5'.format(j+1)
    h5f = h5py.File(path2, 'w')
    h5f.create_dataset('point', data=points_20)
    h5f.create_dataset('label', data=labels_20) 
    h5f.create_dataset('label_name',data=fiber)
    h5f.create_dataset('region',data=regions_20)
    h5f.create_dataset('ind_region',data=ind_regions_20)
    print(regions_20.shape)

  
