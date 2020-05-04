import os

import torch
import numpy as np
import argparse
import time
from lib import utils
from lib.metrics import metrics_torch
from model.pytorch import supervisor
from model.pytorch.engine import Trainer
from model.pytorch.gwnet_model import gwnet
from model.pytorch.lstm_model import LSTMNet
from model.pytorch.supervisor import Supervisor

# basedir is data/metr-la
def save_model(model, basedir, fname):
    path = os.path.join(basedir, os.path.dirname(fname))
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(basedir, fname)
    torch.save(model.state_dict(), fpath)
    print('saved', fpath)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = utils.load_adj(args.adjdata, args.adjtype)
    ds = utils.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    dataloader = ds.data
    num_nodes = dataloader['x_shape'][2]  # [N, seq, nodes, in_dim]

    # Model loading
    print(args)
    sv = Supervisor(adj_mx, args)

    # Training
    scaler = dataloader['scaler']
    # [N, seq, nodes, in_dim]
    num_nodes = dataloader['x_shape'][2]
    engine = Trainer(scaler, device, sv.model, args.learning_rate, args.weight_decay)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        print(dataloader['train_loader'].xs.shape, sv)
        train_loss, train_rmse, train_mape = sv.run_epoch(dataloader['train_loader'], engine.train, args.print_every)
        t2 = time.time()
        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mape = np.mean(train_mape)
        train_time.append(t2 - t1)

        # validation
        s1 = time.time()
        valid_loss, valid_rmse, valid_mape = sv.run_epoch(dataloader['val_loader'], engine.eval, print_every=-1)
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mvalid_loss = np.mean(valid_loss)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mape = np.mean(valid_mape)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Training Time: {:.4f}/epoch'
        print(
            log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mvalid_loss, mvalid_rmse, mvalid_mape, (t2 - t1)),
            flush=True)
        save_model(engine.model, ds.basedir,
                   args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    try:
        engine.model.load_state_dict(
            torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    except:
        print('Reloading model failed!')
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    save_model(engine.model, ds.basedir, args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")

    # Testing
    sv.show_multiple_horizon(scaler, dataloader['test_loader'])


if __name__ == "__main__":
    parser = supervisor.get_argument_parser()
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--save', type=str, default='checkpoints/lstm', help='save path')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument('--print_every',type=int,default=50,help='')
    args = parser.parse_args()
    main(args)