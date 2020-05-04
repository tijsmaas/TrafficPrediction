#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:40:20 2020
1. Load HDF file
3. Enrich samples with time/day of week
2. Data augmentation

Originally from DCRNN utils.py, lib.generate_training_data.py
@author: tijs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import os

import numpy as np
import pandas as pd
import pickle


# Same permutation and augmentation for every model (run)
np.random.seed(seed=42)


class Dataset():
    def __init__(self, dataset_file):
        print ('Loading dataset', dataset_file)
        self.df = pd.read_hdf(dataset_file)
        self.dataset_file = dataset_file
        self.basedir = os.path.dirname(dataset_file)
        self.ds_name = os.path.basename(self.basedir)
        self.data = {}
        # Load the scaler or recompute+store it
        f_scaler = dataset_file + '.scaler.npz'
        try:
            ld = np.load(f_scaler)
            mean, std = float(ld['mean']), float(ld['std'])
            self.scaler = StandardScaler(mean, std)
        except:
            print ('Computing scaler...')
            x_train, y_train, _, _ = self.generate_train_val_test(category='train',
                                                                  add_time_in_day=True,
                                                                  add_day_in_week=False)
            self.scaler = StandardScaler.compute_from(x_train)
            # Ignore scale for timeofday/dayofweek variable
            self.scaler.mean = float(self.scaler.mean[0])
            self.scaler.std = float(self.scaler.std[0])
            # for i in range(self.scaler.mean.shape[0] - 1):
            #     self.scaler.mean[i+1] = 0
            #     self.scaler.std[i+1] = 1.0
            np.savez(f_scaler,
                     mean = self.scaler.mean,
                     std = self.scaler.std)
        self.data['scaler'] = self.scaler

    def load_category(self, category='val', batch_size=64, pad_with_last_sample=False, add_time_in_day=True, add_day_in_week=False):
        x, y, _, _ = self.generate_train_val_test(category,
                                                  add_time_in_day=add_time_in_day,
                                                  add_day_in_week=add_day_in_week)
        # Data format
        x[..., 0] = self.scaler.transform(x[..., 0])
        # y = self.scaler.transform(y)
        loader = DataLoader(x, y, batch_size, pad_with_last_sample=pad_with_last_sample)

        # def __len__(self):
        #     """Denotes the total number of samples"""
        #     return len(self.list_IDs)
        #
        # ''' In python, iteration is build on two primitives, __getitem__() and __iter__().
        # Now, PyTorch requires a dataset to implement __len__ and __getitem__, whereas Tensorflow requires __iter__().
        # We define iter
        #  '''
        # def __getitem__(self, index):
        #     'Generates one sample of data'
        #     # Select sample
        #     ID = self.list_IDs[index]
        #
        #     # Load data and get label
        #     X = torch.load('data/' + ID + '.pt')
        #     y = self.labels[ID]
        #
        #     return X, y

        self.data['x_shape'] = x.shape
        self.data['y_shape'] = y.shape
        self.data['x_' + category] = x
        self.data['y_' + category] = y
        self.data[category + '_loader'] = loader

        # it = loader.get_iterator()
        # len2 = 0
        # try:
        #     while(True):
        #         next(it)
        #         len2 += 1
        # except:
        #     pass
        #
        # print (len(x), len2)
        # assert len(x) == len2
        return loader

    def generate_graph_seq2seq_io_data(self, df,
            x_offsets, y_offsets, skip=1, add_time_in_day=True, add_day_in_week=False
    ):
        """
        Enrich samples with time/day of week, generate samples from:
        :param df:
        :param x_offsets:
        :param y_offsets:
        :param add_time_in_day:
        :param add_day_in_week:
        :return:
        # x: (epoch_size, input_length, num_nodes, input_dim)
        # y: (epoch_size, output_length, num_nodes, output_dim)
        """

        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        feature_list = [data]
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(time_in_day)
        if add_day_in_week:
            dow = df.index.dayofweek
            dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(dow_tiled)

        data = np.concatenate(feature_list, axis=-1)
        # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        for t in range(min_t, max_t, skip):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def generate_train_val_test(self, category='test', add_time_in_day=True, add_day_in_week=False):
        y_start = 1 #args.y_start
        seq_length_x, seq_length_y = 12, 12 #args.seq_length_x, args.seq_length_y

        # 0 is the latest observed sample., (-11, 1, 1)
        x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
        # Predict the next one hour, (1, 13, 1)
        y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

        # x: (num_samples, input_length, num_nodes, input_dim)
        # y: (num_samples, output_length, num_nodes, output_dim)

        # Write the data into npz file.
        # num_test = 6831, using the last 6831 examples as testing.
        # for the rest: 7/8 is used for training, and 1/8 is used for validation.
        train_ratio = 0.7 # args.train_ratio
        test_ratio = 0.2 # args.test_ratio

        sample_padding = seq_length_x + seq_length_y - 1
        num_samples = self.df.shape[0]
        train_num = round(train_ratio * num_samples)
        test_num = round(test_ratio * num_samples)
        val_num = num_samples - train_num - test_num

        # FIXME: Is the order of splitting df a problem?
        if category == 'train':
            df_sub = self.df[:train_num]
        elif category == 'val':
            df_sub = self.df[train_num: train_num + val_num]
        elif category == 'test':
            df_sub = self.df[-test_num:]
        else:
            df_sub = self.df

        x, y = self.generate_graph_seq2seq_io_data(
            df_sub,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            add_time_in_day=add_time_in_day,
            add_day_in_week=add_day_in_week,
        )

        print("x shape: ", x.shape, ", y shape: ", y.shape)
        x_offsets = x_offsets.reshape(list(x_offsets.shape) + [1])
        y_offsets = y_offsets.reshape(list(y_offsets.shape) + [1])

        return x, y, x_offsets, y_offsets

    def get_sensor_coords(self):
        # Prepare sensor coordinates (distance)
        sensor_ids_df = pd.read_csv(os.path.join(self.basedir, 'graph_sensor_locations.csv'),
                                    dtype={'index': 'int', 'sensor_id': 'str',
                                           'latitude': 'float', 'longitude': 'float'})
        sensor_ids = [None] * len(sensor_ids_df.values)
        sensor_coords = [None] * len(sensor_ids_df.values)
        for index, sensor_id, lat, long in sensor_ids_df.values:
            sensor_ids[index] = sensor_id
            sensor_coords[index] = np.array([lat, long])
        return np.array(sensor_coords)

    # e.g. fname='results/lstm_preds'
    def experiment_save(self, pred_mx, fname):
        path = os.path.join('data', self.ds_name, os.path.dirname(fname))
        os.makedirs(path, exist_ok=True)
        fpath = os.path.join('data', self.ds_name, fname)
        np.savez_compressed(fpath, pred_mx)
        print('saved', fname)

    def experiment_save_plot(self, plt, fname='viz/hm.pdf'):
        path = os.path.join('data', self.ds_name, os.path.dirname(fname))
        os.makedirs(path, exist_ok=True)
        fpath = os.path.join('data', self.ds_name, fname)
        plt.savefig(fpath)
        print('saved', fname)


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
        """
        Feeds data to torch/tensorflow
        Also has the ability to augment the data.
        :param xs: every item = [M timesteps history]
        :param ys: every item = [M timesteps future]
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    # Set sensor value to 0
    def augment(self, augmentation_matrix):
        # np.random.seed(seed)
        # s_id = np.random.randint(0, self.xs.shape[2])
        s_id = (augmentation_matrix == 1)
        new_inst = copy.deepcopy(self)
        new_inst.xs[:, :, s_id, 0] = 0
        new_inst.ys[:, :, s_id, 0] = 0
        # print ('Setting', str(s_id), 'to 0')
        return new_inst


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std, fill_zeroes=False):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data[..., 0] == 0)
            data[mask, 0] = self.mean[0]
        return (data - self.mean) / self.std
    #
    # def __init__(self, mean, std):
    #     self.mean = mean
    #     self.std = std
    #
    # def transform(self, data):
    #     return (data - self.mean) / self.std

    def compute_from(x_train, samples_per_hour=12, round_to_days=False, exclude_unk=False):
        if round_to_days:
            # The train set might end with a half day and this could mess with the distribution
            # The daily repetition pattern means that we should compute the scaler
            #  over n days = 24*12 frames per day.
            day = 24*samples_per_hour
            n_days = x_train.shape[0] // day
            print (n_days, 'days, discard timeframes', (x_train.shape[0] - n_days*day))
            x_train = x_train[-n_days*day:,:,:,:]
        # Generate a scaler object using the train set
        dx_train = x_train.reshape(-1, x_train.shape[-1])
        # Trick to exclude unknown (0) values from mean
        if exclude_unk:
            dx_train[dx_train == 0] = np.nan
        mean, std = np.nanmean(dx_train, axis=0), np.nanstd(dx_train, axis=0)
        scaler = StandardScaler(mean=mean, std=std)
        return scaler


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

# Orig graph wavenet
def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        print('UnicodeDecodeError, reloading ', pickle_file, 'with enc=latin1')
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    print ('Loading dataset')
    ds = Dataset(args.traffic_df_filename)
    print("Generating training data")
    ds.generate_train_val_test()
    print (ds.scaler.mean)
    val_loader = ds.load_category(category='test')
    x_val, y_val = val_loader.xs, val_loader.ys

