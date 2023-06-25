from load_data import TestAllDataset
import torch
import argparse
from model import ResPointNetCls
import os
import time
import h5py
import numpy as np
use_cpu = True
device = torch.device("cpu")
# if use_cpu:
#     device = torch.device("cpu")
# else:
#     device = torch.device("cuda:0")
parser = argparse.ArgumentParser(description="classfy SWM and DWM on test dataset")
parser.add_argument('--test_batch_size',type=int,default=512)
parser.add_argument('--best_metric',type=str,default='f1')
args = parser.parse_args()
out_path = './ModelWeights/nc_ad_mc/'  #classify SWM and DWM
weight_path = 'PretrainedModel/TrainedModel_TwoStage/s1_cls'  #the model from SupWMA
if not os.path.exists(out_path):
    os.makedirs(out_path)
k = 800

def load_model():
    classifer = ResPointNetCls(k=k).to(device)
    classifer_weight_path = os.path.join(weight_path,'best_{}_model.pth'.format(args.best_metric))
    classifer.load_state_dict(torch.load(classifer_weight_path))
    return classifer
def test_net():
    print('')
    print('===================================')
    print('')
    classifer_net = load_model()
    start_time = time.time()
    with torch.no_grad():
        test_predicted_lst = []
        for j,data in (enumerate(test_data_loader, 0)):
            points = data
            points = points.transpose(2, 1)  #(10000,15,3)-->(10000,3,15) (512,3,15) regions:(512,)
            points = points.to(device)
            classifer_net = classifer_net.eval()
            pred = classifer_net(points)
            _,pred_idx = torch.max(pred, dim=1)
            pred_idx = torch.where(pred_idx < k, pred_idx, torch.tensor(k).to(device))
            pred_idx = pred_idx.cpu().detach().numpy().tolist()
            test_predicted_lst.extend(pred_idx)
    end_time = time.time()
    print('The total time of prediction is:{} s'.format(round((end_time - start_time), 4)))
    print('The test sample size is: ', len(test_predicted_lst))
    test_prediction_lst_h5 = h5py.File(output_prediction_mask_path,'w')
    test_prediction_lst_h5['complete_pred_test'] = test_predicted_lst
    test_predicted_array = np.asarray(test_predicted_lst)
    return test_predicted_array

for i in range(125):  #125 subjects in test dataset
    test_dataset = TestAllDataset(num=i)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = args.test_batch_size,shuffle=False,num_workers=4)
    test_data_size = len(test_dataset)
    print('test data size : {}'.format(test_data_size))
    # num_classes = len(test_dataset.label_names)
    num_classes = k
    output_prediction_mask_path = os.path.join(out_path + 'test_orediction_mask_{}.h5'.format(i))
    # test_data_loader, num_class = load_test_data()
    predicted_arr = test_net()
