import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


# RNN model for every sensor
class LSTMNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, one_lstm=True):
        super(LSTMNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.one_lstm = one_lstm
        self.hidden = residual_channels
        self.receptive_field = 1
        self.device = device

        if one_lstm:
            self.rnn_all = nn.LSTM(in_dim * num_nodes, self.hidden, layers, dropout=dropout).to(device)
            self.fcn_all = torch.nn.Sequential(
                torch.nn.Linear(self.hidden, in_dim * num_nodes),
            ).to(device)
        else:
            self.rnn = [nn.LSTM(in_dim, self.hidden, layers, dropout=dropout).to(device) for _ in range(num_nodes)]
            self.fcn = [torch.nn.Sequential(
                torch.nn.Linear(self.hidden, in_dim),
            ).to(device) for _ in range(num_nodes)]

        self.timeframes = out_dim + self.receptive_field
        self.out_dim = out_dim

    def forward(self, input):
        # input: [batch, vals, sensors, measurements]
        skip = 0
        batch = input.size(0) # 64
        vals = input.size(1) # 2
        sensors = input.size(2) #207
        in_len = input.size(3)

        # Padding to compensate for field of view, counts as <EOS>
        x = nn.functional.pad(input, (self.receptive_field,0,0,0))

        timeframes = x.size(3)

        h0 = torch.randn(self.layers, timeframes, self.hidden).cuda() # layers, .., hidden
        c0 = torch.randn(self.layers, timeframes, self.hidden).cuda()

        if self.one_lstm:
            # Encoder
            # LSTM-FC torch.Size([64, 12, 414])
            all_sensors_input = x.view(batch, vals * sensors, -1).transpose(1, 2)
            # Run rnn for every timeseries step: [batch, timeseries, vals*sensors]
            output, (hn, cn) = self.rnn_all(all_sensors_input, (h0, c0))
            # Last predicted char is first prediction
            decoder_inp_hidd = output[:, -1].view(batch, -1, self.hidden)

            # Decoder
            decoder_output = torch.zeros(batch, self.out_dim, sensors*vals).to(self.device)
            hdec = hn[:, -1, :].contiguous().view(vals, 1, -1)
            cdec = cn[:, -1, :].contiguous().view(vals, 1, -1)
            for t in range(self.out_dim):
                decoder_inp_char = self.fcn_all(decoder_inp_hidd)
                decoder_pred_char, (hdec, cdec) = self.rnn_all(decoder_inp_char, (hdec, cdec))
                decoder_output[:, t:t+1, :] = decoder_inp_char
                decoder_inp_char = decoder_pred_char

            # WARN: splits the linear layer, making half of the weights useless -> [batch, 1, timesteps, sensors]
            output = decoder_output[:, :, 0:sensors].view(batch, self.out_dim, sensors, 1)
            return output

        else:
            decoder_output = torch.zeros(batch, self.out_dim, sensors, vals).to(self.device)
            # make prediction for each sensor individually
            c = torch.chunk(x, sensors, dim=2)
            for idx, chunk in enumerate(c):
                single_sensor_input = chunk.squeeze()
                single_sensor_input_sw = single_sensor_input.transpose(1, 2)

                # Encoder: Run rnn for every timeseries step
                h0 = torch.randn(self.layers, timeframes, self.hidden).to(self.device)
                c0 = torch.randn(self.layers, timeframes, self.hidden).to(self.device)
                output, (hn, cn) = self.rnn[idx](single_sensor_input_sw, (h0, c0))
                # Last predicted char is first prediction
                decoder_inp_hidd = output[:, -1].view(batch, -1, self.hidden)

                # Decoder
                hdec = hn[:, -1, :].contiguous().view(vals, 1, -1)
                cdec = cn[:, -1, :].contiguous().view(vals, 1, -1)
                for t in range(self.out_dim):
                    decoder_inp_char = self.fcn[idx](decoder_inp_hidd)
                    decoder_pred_char, (hdec, cdec) = self.rnn[idx](decoder_inp_char, (hdec, cdec))
                    decoder_output[:, t:t + 1, idx:idx+1, :] = decoder_inp_char.unsqueeze(1)
                    decoder_inp_char = decoder_pred_char

            # WARN: splits the linear layer, making half of the weights useless -> [batch, 1, timesteps, sensors]
            output = decoder_output[:, :, :, 0:1].view(batch, self.out_dim, sensors, 1)
            return output



    @classmethod
    def from_args(cls, args, device, supports, aptinit, **kwargs):
        defaults = dict(dropout=args.dropout, supports=supports,
                        addaptadj=args.addaptadj, aptinit=aptinit,
                        in_dim=args.in_dim, out_dim=args.seq_length,
                        residual_channels=args.nhid, dilation_channels=args.nhid,
                        one_lstm=not args.isolated_sensors)
        defaults.update(**kwargs)
        model = cls(device, args.num_sensors, **defaults)
        return model

    def load_checkpoint(self, state_dict):
        # only weights that do *NOT* depend on seq_length
        self.load_state_dict(state_dict, strict=False)