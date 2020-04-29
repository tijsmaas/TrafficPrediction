import numpy as np

# log string
from lib.dataloaders.dataloader import Dataset

def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros(shape = (num_sample, num_his, dims))
    y = np.zeros(shape = (num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i : i + num_his]
        y[i] = data[i + num_his : i + num_his + num_pred]
    return x, y

def loadData(args):
    ds = Dataset(args.traffic_file)
    ds.load_category('train', args.batch_size, add_time_in_day=False, add_day_in_week=False)
    ds.load_category('val', args.batch_size, add_time_in_day=False, add_day_in_week=False)
    ds.load_category('test', args.batch_size, add_time_in_day=False, add_day_in_week=False)

    # train/val/test
    sample_padding = args.num_his + args.num_pred - 1
    train_steps = ds.data['x_train'].shape[0] + sample_padding # 23990
    val_steps = ds.data['x_val'].shape[0] + sample_padding  # 3428
    test_steps = ds.data['x_test'].shape[0] + sample_padding # 8654
    # X, Y
    trainX, trainY = np.squeeze(ds.data['x_train'],axis=3), np.squeeze(ds.data['y_train'],axis=3)
    valX, valY = np.squeeze(ds.data['x_val'],axis=3), np.squeeze(ds.data['y_val'],axis=3)
    testX, testY = np.squeeze(ds.data['x_test'],axis=3), np.squeeze(ds.data['y_test'],axis=3)
    mean, std = ds.scaler.mean, ds.scaler.std

    # spatial embedding 
    f = open(args.SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    num_vertex, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (num_vertex, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
        
    # temporal embedding 
    Time = ds.df.index
    dayofweek = np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, num_his + num_pred, 2)
    trainTE = seq2instance(train, args.num_his, args.num_pred)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)

    # Authors consider time embedding (TE) different from input (X) and labels (Y), we follow their work.
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std, ds)
