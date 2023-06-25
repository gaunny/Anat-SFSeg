#data registered to ORG atlas, project them to Freesurfer parcellation to obtain individual-level anatomical information
import vtk
from vtkmodules.vtkCommonCore import vtkIdList
import numpy as np
import re
import glob
from os.path import join, exists
import nibabel
from functools import reduce
from nibabel.affines import apply_affine
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

arender = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(arender)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
reader = vtk.vtkPolyDataReader()

vtk_all_files = natural_sort(glob.glob('./data/AD_MCI_S01/*/FiberBundle_reg_reg.vtk'))

numberoflines_all = 0
for i in range(len(vtk_all_files)):
    name = vtk_all_files[i].split('/')[-2]
    print(name) 
    vtk_file = './data/AD_MCI_S01/{}/FiberBundle_reg_reg.vtk'.format(name)
    reader.SetFileName(vtk_file)  
    reader.Update()
    inpd = reader.GetOutput()
    inpoints = inpd.GetPoints()
    inpd.GetLines().InitTraversal()
    # read lines
    numberOfLines = inpd.GetNumberOfLines()
    print('There are {0} lines in the polydata'.format(numberOfLines))
    point_fiber_all = None
    for lidx in range(0, numberOfLines):  #fiber
        indices = vtk.vtkIdList()
        inpd.GetLines().GetNextCell(indices)
        point_all = None
        for i in range(indices.GetNumberOfIds()):  #point
            
            point = inpoints.GetPoint(indices.GetId(i))
            point_array = np.array(point)  #(3,)
            # point_list.append(point)
            if i == 0:
                point_array = point_array[np.newaxis,:]
                point_all = point_array
            else:
                point_array = point_array[np.newaxis,:]
                point_all = np.vstack((point_all,point_array)) 
        ind = np.arange(indices.GetNumberOfIds()) 
        if ind.shape[0] >14:  #sample to 15
            sub_ind = np.random.choice(ind, 15, replace=False)
            arrIndex = np.array(sub_ind).argsort()  
            sub_ind_sort = sub_ind[arrIndex]
        else:
            sub_ind = np.random.choice(ind, 15, replace=True)
            arrIndex = np.array(sub_ind).argsort()
            sub_ind_sort = sub_ind[arrIndex]
        sub_points = np.array(point_all)[sub_ind_sort]   #(15,3)
        if lidx == 0:
            sub_points = sub_points[np.newaxis,:]
            point_fiber_all = sub_points
        else:
            sub_points = sub_points[np.newaxis,:]
            point_fiber_all = np.vstack((point_fiber_all,sub_points))
    print(point_fiber_all.shape)  #(linesnumber,15,3)
    sub_line = np.random.choice(point_fiber_all.shape[0],10000,replace=False)
    line_index = np.array(sub_line).argsort()
    sub_line_sort = sub_line[line_index]
    sub_lines = np.array(point_fiber_all)[sub_line_sort]
    np.save('./data/AD_MCI_S01/{}/sub_tract.npy'.format(name),sub_lines)   #(10000,15,3)
    print(sub_lines.shape)
    refvolume = './data/AD_MCI_S01/{}/parcellation/parcellation/mri/aparc+aseg.nii.gz'.format(name)
    volume = nibabel.load(refvolume)
    volume_shape = volume.get_fdata().shape #(260,311,260)
    volume_content = volume.get_fdata().astype(int)
    label_set = set(np.unique(volume_content))
    label_list = list(map(int,label_set))
    len_label_list = len(label_list)
    down_features = sub_lines #(10000,15,3)

    x_all = None
    point_ijk = apply_affine(np.linalg.inv(volume.affine), down_features)
    
    point_ijk = np.rint(point_ijk).astype(np.int32)
    regions = []
    for i in range(10000):
        point_list = [(point_ijk[i,j,0], point_ijk[i,j,1], point_ijk[i,j,2]) for j in range(point_ijk.shape[1])]
        point_list = set(point_list)
        region = np.zeros((len_label_list,), dtype=int)
        for x, y, z in list(point_list):
            c_idx = volume_content[x, y, z]
            c_idx_ = label_list.index(c_idx)
            # print(c_idx_)
            region[c_idx_] += 1
        regions.append(region)
        # print(len(regions[0]))
        x = np.vstack((region,label_list))
        if i == 0:
            x = x[np.newaxis,:]
            x_all = x
        else:
            x = x[np.newaxis,:]
            x_all = np.vstack((x_all,x))
    print('x_all',x_all.shape)
    np.save('./data/AD_MCI_S01/{}/num_region.npy'.format(name),x_all)


#delete cerebral white matter regions

paths = natural_sort(glob.glob('./data/AD_MCI_S01/*/num_region.npy'))
sort_k_label_list = []
content_list = []
shuffle_num_list = []
for k in range(len(paths)):
    id_name = paths[k].split('/')[-2]
    content = np.load(paths[k])
    arrIndex = np.array(content[0,1,:]).argsort()
    sort_k_label = list(content[0,1,:][arrIndex])
    sort_k_label_list.append(sort_k_label)
    content_list.append(content)

re = reduce(np.intersect1d,[sort_k_label_list[k] for k in range(len(sort_k_label_list))])
re = list(re)
re.remove(0)  #delete 0，2，41,77
re.remove(2)
re.remove(41)
re.remove(77)

all_list = content_list
a_new_all = None
all = None
for j in range(len(paths)):
    id_name = paths[j].split('/')[-2]
    for i in range(len(re)):
        index = np.where(all_list[j][0,1,:]==re[i])
        # a_new_label = a[:,1,:][index]
        a_new_num = all_list[j][:,0,index]
        a_new_label = all_list[j][:,1,index]
        a_new = np.concatenate((a_new_num,a_new_label),axis=1)
        if i == 0:
            a_new_all = a_new
        else:
            a_new_all = np.concatenate((a_new_all,a_new),axis=2)
    print(a_new_all.shape)  #(10000,2,105)
    np.save('./data/AD_MCI_S01/{}_105regions.npy'.format(id_name),a_new_all)
    if j == 0 :
        a_new_all = a_new_all[np.newaxis,:]
        all = a_new_all
    else:
        a_new_all = a_new_all[np.newaxis,:]
        all = np.vstack((a_new_all,all))
print(all.shape)
np.save('./data/AD_MCI_S01/all_ad_num_region_avg_105npy',all)
    