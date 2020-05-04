import torch
import numpy as np
import argparse
import time
from lib import utils
from lib.metrics import metrics_torch, metrics_np
from model.pytorch.engine import Trainer, Evaluator
from model.pytorch.gwnet_model import gwnet
from model.pytorch.lstm_model import LSTMNet


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default='data/metr-la/metr-la.h5', help='data path')
    parser.add_argument('--adjdata', type=str, default='data/metr-la/adj_mx.pkl', help='adj data path')
    parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
    parser.add_argument('--seq_length', type=int, default=12, help='')
    parser.add_argument('--nhid', type=int, default=32, help='')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # model selection
    parser.add_argument('--checkpoint', type=str, help='load a model')
    parser.add_argument('--isolated_sensors', action='store_true', help='separate model for each sensor')
    parser.add_argument('--lstm', action='store_true', help='whether to choose the lstm model instead')

    return parser


class Supervisor:

    def __init__(self, adj_mx, args):
        self.device = args.device
        self.args = args
        self.model = self.load_model(adj_mx, args)


    # datapair from loader -> model
    def _prepare_data(self, x, y):
        x = torch.Tensor(x).to(self.device)
        y = torch.Tensor(y).to(self.device)
        x = x.transpose(1, 3)
        y = y.transpose(1, 3)
        y = y[:, 0, :, :]
        return x, y


    def run_epoch(self, loader, engine_fn, print_every=-1):
        ep_loss = []
        ep_rmse = []
        ep_mape = []
        for iter, (x, y) in enumerate(loader.get_iterator()):
            x, y = self._prepare_data(x, y)
            mae, rmse, mape = engine_fn(x, y)
            ep_loss.append(mae)
            ep_rmse.append(rmse)
            ep_mape.append(mape)
            if iter % print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}'
                print(log.format(iter, ep_loss[-1], ep_rmse[-1], ep_mape[-1]), flush=True)
            # if iter > 50: return
        return ep_loss, ep_rmse, ep_mape


    def load_model(self, adj_mx, args):
        assert adj_mx is not None and len(adj_mx) > 0, "We use the adjacency matrix to initialise the number of nodes in the model"
        num_nodes = len(adj_mx[0])
        supports = [torch.tensor(i).to(self.device) for i in adj_mx]

        if args.randomadj:
            adjinit = None
        else:
            adjinit = supports[0]

        if args.aptonly:
            supports = None

        if args.lstm:
            print('Selected LSTM-FC model')
            # --nhid 256 --weight_decay 0.0005 --learning_rate 0.001 --isolated_sensors False -checkpoint data/metr-la/pretrained/fc_lstm.pth
            model = LSTMNet(self.device, num_nodes, args.dropout, supports=0, gcn_bool=args.gcn_bool,
                          addaptadj=args.addaptadj, aptinit=0, in_dim=args.in_dim, out_dim=args.seq_length,
                          residual_channels=args.nhid,
                          dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16)
            model.to(self.device)
            if args.checkpoint:
                model.load_checkpoint(torch.load(args.checkpoint))
        else:
            # --gcn_bool --addaptadj --checkpoint data/metr-la/pretrained/graph_wavenet_repr.pth
            print('Selected Graph Wavenet model')
            model = gwnet(self.device, num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                          addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length,
                          residual_channels=args.nhid,
                          dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16)
            model.to(self.device)
            if args.checkpoint:
                model.load_state_dict(torch.load(args.checkpoint))
        print('model loaded successfully')
        return model


    def show_multiple_horizon(self, scaler, loader):
        args = self.args
        engine = Evaluator(scaler, self.device, self.model)

        yhat, realy = engine.compute_preds(loader)
        amae = []
        armse = []
        amape = []
        for i in range(args.seq_length):
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            mae, rmse, mape = metrics_np.calculate_metrics(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            armse.append(rmse)
            amape.append(mape)

        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape)))

        return yhat, realy
