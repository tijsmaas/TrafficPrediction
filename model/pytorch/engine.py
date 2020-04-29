import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from lib.metrics import metrics_torch
from model.pytorch.gwnet_model import gwnet

class Evaluator():
    def __init__(self, scaler, device, model):
        self.model = model
        self.model.to(device)
        self.scaler = scaler
        self.device = device

    def compute_preds(self, loader):
        self.model.eval()
        outputs = []
        y_vals = []
        for iter, (x, y) in tqdm(enumerate(loader.get_iterator())):
            testx = torch.Tensor(x).to(self.device)
            testy = torch.Tensor(y[..., 0]).to(self.device)
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                # [64, 12, 1, 207]
                output = self.model(testx)
            preds = output.transpose(1, 3)
            outputs.append(preds.squeeze())
            y_vals.append(testy.transpose(1, 2))
        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(y_vals, dim=0)
        return yhat.cpu().numpy(), realy.cpu().numpy()

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        with torch.no_grad():
            output = self.model(input)
        predict, real = self._postprocess(output, real_val)
        loss, rmse, mape = metrics_torch.calculate_metrics_torch(predict, real, 0.0)
        return loss.item(), rmse.item(), mape.item()

    # model -> evaluation
    # output = [batch_size,12,num_nodes,1]
    def _postprocess(self, output, real_val):
        output = output.transpose(1,3)
        predict = self.scaler.inverse_transform(output)
        real = torch.unsqueeze(real_val, dim=1)
        return predict, real


class Trainer(Evaluator):
    def __init__(self, scaler, device, model, lrate, wdecay):
        super().__init__(scaler, device, model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = metrics_torch.masked_mae_torch
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        predict, real = self._postprocess(output, real_val)
        loss, rmse, mape = metrics_torch.calculate_metrics_torch(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item(), rmse.item(), mape.item()

