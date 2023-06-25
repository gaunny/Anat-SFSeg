#here we show the visualization of SWM and DWM for clsSWM&DWM.py 
import h5py
import numpy as np
import vtk
import re
import glob
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  #isdigt()方法字符串是否全为数字，若全是数字，为True，否则为Fasle.Python lower() 方法转换字符串中所有大写字符为小写
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
#classfy swm&dwm
points_path = natural_sort(glob.glob('./data/AD_MCI_S01/*/sub_tract.npy'))
regions_all = np.load('./data/AD_MCI_S01/all_ad_num_region_avg_105npy.npy')
print(regions_all.shape)  #(125,10000,2,105)
    
for i in range(len(points_path)):
    id = points_path[i].split('/')[-2]
    print(id)
    h5_path = './data/AD_MCI_S01/test_orediction_mask_{}.h5'.format(i)
    h5_feat = h5py.File(h5_path,'r')
    test_predicted_lst = np.asarray(h5_feat['complete_pred_test'])
    points = np.load(points_path[i])
    ind_regions = regions_all[i,:,0,:]
    names = list(range(0,800,1))
    y_pred = list(test_predicted_lst)
    target_names = [str(x) for x in names]
    point_swm = None 
    point_dwm = None
    ind_region_swm = None
    ind_region_dwm = None
    i_swm_lst = []
    for i in range(198):
        index = np.where(test_predicted_lst==i)[0]
        point_i = points[index]
        point_i = np.around(point_i,decimals=4)
        ind_region = ind_regions[index]
        if point_i.shape[0]!=0:
            i_swm = i
            i_swm_lst.append(i_swm)
        if point_i.shape[0]!=0:
            if i ==i_swm_lst[0]:
                point_swm = point_i
                ind_region_swm = ind_region
            else:
                point_swm = np.concatenate((point_swm,point_i))
                ind_region_swm = np.concatenate((ind_region_swm,ind_region))
    i_dwm_lst = []
    for j in range(198,800):
        index = np.where(test_predicted_lst==j)[0]
        point_i = points[index]
        point_i = np.around(point_i,decimals=4)
        ind_region = ind_regions[index]
        if point_i.shape[0]!=0:
            i_dwm = j
            i_dwm_lst.append(i_dwm)
        if point_i.shape[0]!=0:
            if j == i_dwm_lst[0]:
                point_dwm = point_i
                ind_region_dwm = ind_region
            else:
                point_dwm = np.concatenate((point_dwm,point_i))
                ind_region_dwm = np.concatenate((ind_region_dwm,ind_region))
    np.save('./ModelWeights/nc_ad_mc/point_swm_{}.npy'.format(id),point_swm)      
    np.save('./ModelWeights/nc_ad_mc/point_dwm_{}.npy'.format(id),point_dwm)    
    np.save('./ModelWeights/nc_ad_mc/105region_swm_{}.npy'.format(id),ind_region_swm)      
    np.save('./ModelWeights/nc_ad_mc/105region_dwm_{}.npy'.format(id),ind_region_dwm)    

    print(point_swm.shape)
    print(point_dwm.shape)
    # print(ind_region_dwm.shape)
    lines = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    # save to .vtk
    for i in range(point_swm.shape[0]):
        line = vtk.vtkPolyLine()
        for j in range(point_swm.shape[1]):
            point_id = points.InsertNextPoint(point_swm[i, j])
            line.GetPointIds().InsertNextId(point_id)
        lines.InsertNextCell(line)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("./ModelWeights/nc_ad_mc/cluster_swm_{}.vtk".format(id))
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()


    lines = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    for i in range(point_dwm.shape[0]):
        line = vtk.vtkPolyLine()
        for j in range(point_dwm.shape[1]):
            point_id = points.InsertNextPoint(point_dwm[i, j])
            line.GetPointIds().InsertNextId(point_id)
        lines.InsertNextCell(line)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("./ModelWeights/nc_ad_mc/cluster_dwm_{}.vtk".format(id))
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()
