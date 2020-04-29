import torch
from tqdm import tqdm

from lib import utils
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model.pytorch.gwnet_model import gwnet
from model.pytorch.lstm_model import LSTMNet

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/metr-la/metr-la.h5',help='path to h5 dataset')
parser.add_argument('--adjdata',type=str,default='data/metr-la/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--plotheatmap',type=str,default='True',help='')

parser.add_argument('--lstm',action='store_true',help='whether to choose the lstm model instead')


args = parser.parse_args()


def compute_preds(scaler, loader, model, device):
    outputs = []
    y_vals = []
    for iter, (x, y) in enumerate(loader.get_iterator()):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y[..., 0]).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            # [64, 12, 1, 207]
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())
        y_vals.append(testy.transpose(1, 2))
    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(y_vals, dim=0)
    return yhat.cpu().numpy(), realy.cpu().numpy()

def main():
    device = torch.device(args.device)

    _, _, adj_mx = utils.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    if args.lstm:
        print('Selected LSTM-FC model')
        # --device cuda:0 --nhid 256 --weight_decay 0.0005 --learning_rate 0.001 --isolated_sensors False --num_sensors 207   --checkpoint data/metr-la/pretrained/graph_wavenet_repr.pth
        args.nhid = 256
        args.weight_decay = 0.0005
        args.learning_rate = 0.001
        args.num_sensors = 207
        args.isolated_sensors = False
        model = LSTMNet.from_args(args, device, supports=0, aptinit=0)
        model.to(device)
        if args.checkpoint:
            model.load_checkpoint(torch.load(args.checkpoint))
    else:
        # --device cuda:0 --gcn_bool --addaptadj --checkpoint data/metr-la/pretrained/graph_wavenet_repr.pth
        print('Selected Graph Wavenet model')
        model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit)
        model.to(device)
        model.load_state_dict(torch.load(args.checkpoint))

    model.eval()
    print('model load successfully')

    print('Evaluating with simulated sensor failure...')

    ds = utils.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    dataloader = ds.data
    scaler = dataloader['scaler']
    category = 'val'
    loader = dataloader[category + '_loader']
    preds, realy = compute_preds(scaler, loader, model, device)
    print (preds.shape, realy.shape)
    # Augment for 12 sensors on the map
    # dis_sensors = [3, 4, 5, 6, 12, 15, 16, 17, 23, 26, 29, 30, 33, 38, 48, 56, 64, 65, 80, 91, 93, 101, 124, 133, 134,
    #                136, 138, 144, 154, 155, 157, 159, 160, 161, 162, 163, 165, 166, 170, 174, 187, 188, 191, 192, 193,
    #                195, 196]
    dis_sensors = range(206) #[189, 200, 18, 35, 50, 21, 121, 189, 126]

    # Augmentation pattern, disable all sensors individually
    all_preds = []
    for idx, s in tqdm(enumerate(dis_sensors)):
        augmentation_matrix = np.zeros(207)
        augmentation_matrix[s] = 1

        # Generate augmented datasets
        augmented_dataloader = loader.augment(augmentation_matrix)
        # Do inference [3392, 207, 12]
        aug_preds, _ = compute_preds(scaler, augmented_dataloader, model, device)
        # relative_err = MAPE(normal) - MAPE(augmented)
        #  mae error per sensor before - error per sensor after
        err_rel = (realy != 0).astype(np.uint8) * (np.abs(preds - realy) - np.abs(aug_preds - realy))
        # Scale relative_err per 'frame' with softmax, then average over time.
        all_preds.append(err_rel)
        print (err_rel.shape)

    pred_mx = np.stack(all_preds)
    # Aggregate over time
    pred_mx = np.sum(pred_mx, axis=1) / pred_mx.shape[1]
    print(pred_mx.shape)
    if args.lstm:
        ds.experiment_save(pred_mx, 'results/lstm_preds')
    else:
        ds.experiment_save(pred_mx, 'results/graph_wavenet_preds')

    # Heatmap
    plot = np.sum(pred_mx[:, dis_sensors, ...], axis=2)
    for i in range(plot.shape[0]):
        plot[i, i] = 0.0
    sns.heatmap(plot, cmap="RdYlBu")
    ds.experiment_save_plot(plt, 'viz/hm.pdf')

if __name__ == "__main__":
    main()
