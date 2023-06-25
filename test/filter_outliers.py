import numpy as np
import argparse
import h5py
import time
import os
import pickle
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from model_supcon import PointNet_SupCon, PointNet_Classifier
from load_data import TestSWMDataset

parser = argparse.ArgumentParser(description="filter outliers on SWM from stage one")
parser.add_argument('--weight_path', type=str, help='pretrained network model')
args = parser.parse_args()
k=198

def load_model():
    #stage two model from SupWMA
    stage2_encoder = PointNet_SupCon(head=encoder_params['head_name'], feat_dim=encoder_params['encoder_feat_num']).to(device)
    stage2_classifer = PointNet_Classifier(num_classes=encoder_params['stage2_num_class']).to(device)
    # load weights
    encoder_weight_path = os.path.join(args.weight_path, 's2_encoder', 'epoch_100_model.pth')
    stage2_encoder.load_state_dict(torch.load(encoder_weight_path))
    classifier_weight_path = os.path.join(args.weight_path, 's2_cls', 'best_f1_model.pth')
    stage2_classifer.load_state_dict(torch.load(classifier_weight_path))
    return stage2_encoder, stage2_classifer

def test_net():
    stage2_encoder_net, stage2_classifer_net = load_model()
    if not os.path.exists(output_prediction_mask_path):
        # Load model
        start_time = time.time()
        with torch.no_grad():
            test_predicted_lst = []
            for j, data in (enumerate(test_data_loader, 0)):
                swm_points=data
                swm_points = swm_points.transpose(2, 1)
                swm_points = swm_points.to(device)
                stage2_encoder_net, stage2_classifer_net = \
                    stage2_encoder_net.eval(), stage2_classifer_net.eval()
                features = stage2_encoder_net.encoder(swm_points)
                stage2_pred = stage2_classifer_net(features)
                _, stage2_pred_idx = torch.max(stage2_pred, dim=1)
                stage2_pred_idx = torch.where(stage2_pred_idx < k, stage2_pred_idx, torch.tensor(k).to(device))
                
                stage2_pred_idx = stage2_pred_idx.cpu().detach().numpy().tolist()
                test_predicted_lst.extend(stage2_pred_idx)
        end_time = time.time()
        print('The total time of prediction is:{} s'.format(round((end_time - start_time), 4)))
        print('The test sample size is: ', len(test_predicted_lst))
        
        test_prediction_lst_h5 = h5py.File(output_prediction_mask_path, "w")
        test_prediction_lst_h5['complete_pred_test'] = test_predicted_lst
        test_predicted_array = np.asarray(test_predicted_lst)

    else:
        test_prediction_h5 = h5py.File(output_prediction_mask_path, "r")
        test_predicted_array = np.asarray(test_prediction_h5['complete_pred_test'])

    return test_predicted_array

if __name__ == "__main__":
    use_cpu = False
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    with open(os.path.join(args.weight_path, 's1_cls', 'stage1_params.pickle'), 'rb') as f:
        stage1_params = pickle.load(f)
        f.close()
    with open(os.path.join(args.weight_path, 's2_encoder', 'encoder_params.pickle'), 'rb') as f:
        encoder_params = pickle.load(f)
        f.close()

    for i in range(125):  #125
        test_dataset = TestSWMDataset(num=i)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size = args.test_batch_size,shuffle=False,num_workers=int(args.num_workers))

        test_data_size = len(test_dataset)
        print('test data size : {}'.format(test_data_size))
        # num_classes = len(test_dataset.label_names)
        num_classes = k
        output_prediction_mask_path = './ModelWeights/nc_ad_mc/test_orediction_mask_{}.h5'
    # print(output_prediction_mask_path)
        # test_data_loader, num_class = load_test_data()
        predicted_arr = test_net()

# CUDA_VISIBLE_DEVICES=3 python test_twostage.py --weight_path /TrainedModel_TwoStage