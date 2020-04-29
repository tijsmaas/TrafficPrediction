import torch

from lib import utils
from lib.dataloaders.dataloader import Dataset
from lib.metrics import metrics_torch, metrics_np
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.pytorch import supervisor
from model.pytorch.engine import Evaluator
from model.pytorch.gwnet_model import gwnet
from model.pytorch.lstm_model import LSTMNet
import torch.nn.functional as F

from model.pytorch.supervisor import Supervisor


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = utils.load_adj(args.adjdata, args.adjtype)
    ds = Dataset(args.data)
    ds.load_category('test', args.batch_size)
    dataloader = ds.data
    scaler = dataloader['scaler']

    # Model loading
    print(args)
    sv = Supervisor(adj_mx, args)

    # Testing
    yhat, realy = sv.show_multiple_horizon(scaler, dataloader['test_loader'])


    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(sv.model.nodevec1, sv.model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        ds.experiment_save_plot(plt, 'viz/gwnet_emb.pdf')

    y12 = realy[:,99,11].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[:,99,11]).cpu().detach().numpy()

    y3 = realy[:,99,2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[:,99,2]).cpu().detach().numpy()

    df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real3': y3, 'pred3':yhat3})
    df2.to_csv('./wave.csv',index=False)


if __name__ == "__main__":
    parser = supervisor.get_argument_parser()
    parser.add_argument('--plotheatmap', type=str, default='True', help='')
    args = parser.parse_args()
    main(args)
