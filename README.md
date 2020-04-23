# Spatio-Temporal Traffic Prediction
Preliminary framework to use the same dataloader and evaluation methods for Deep Learning Spatio-Temporal Traffic Prediction. This enables data augmentation and other evaluation methods (e.g.  test model robustness to sensor failure) . 

This repository contains the following implementations:
- (DCRNN) Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
- (Graph Wavenet) Graph WaveNet for Deep Spatial-Temporal Graph Modeling
- (GMAN) Graph Multi-Attention Network for Traffic Prediction


TensorFlow implementation of Diffusion Convolutional Recurrent Neural Network in the following paper: \
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926), ICLR 2018.

PyTorch implementation of Graph WaveNet in the following paper: \
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).

TensorFlow implementation of Graph Multi-Attention Network in the following paper: \
Chuanpan Zheng, Xiaoliang Fan, Cheng Wang, and Jianzhong Qi. "[GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://arxiv.org/abs/1911.08415)", AAAI2020 (https://arxiv.org/abs/1911.08415)


## Tasks
- [X] Add DCRNN
- [X] Add Graph Wavenet
- [X] Add GMAN
- [ ] Add ST-GCN
- [ ] Add [DCRNN-Pytorch](https://github.com/chnsh/DCRNN_PyTorch)
- [ ] Add PEMS-BAY dataset
- [ ] Create performance table (auto-generated)


## Requirements
- scipy>=1.2.0
- numpy>=1.12.1
- pandas>=0.24.2
- pyaml
- statsmodels
- tqdm
- tensorflow>=1.14.0
- torch>=1.4.0

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), and should be
put into the `data/{metr-la|pems-bay}/` directory.
The `*.h5` files store the data in `panads.DataFrame` using the `HDF5` file format. Here is an example:

|                     | sensor_0 | sensor_1 | sensor_2 | sensor_n |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|
| 2018/01/01 00:00:00 |   60.0   |   65.0   |   70.0   |    ...   |
| 2018/01/01 00:05:00 |   61.0   |   64.0   |   65.0   |    ...   |
| 2018/01/01 00:10:00 |   63.0   |   65.0   |   60.0   |    ...   |
|         ...         |    ...   |    ...   |    ...   |    ...   |


Here is an article about [Using HDF5 with Python](https://medium.com/@jerilkuriakose/using-hdf5-with-python-6c5242d08773).

## Run the Pre-trained Model on METR-LA

```bash
# METR-LA
python dcrnn_test.py --config_filename=data/metr-la/pretrained/dcrnn_config.yaml
python gwnet_test.py --checkpoint data/metr-la/pretrained/graph_wavenet_repr.pth --data data/metr-la/metr-la.h5
python gwnet_test.py --lstm --checkpoint data/metr-la/models/fc_lstm.pth --data data/metr-la/metr-la.h5
python gman_train.py --max_epoch 0 --SE_file data/metr-la/SE(METR-LA).txt --model_file data/metr-la/pretrained/GMAN_METR-LA --traffic_file data/metr-la/metr-la.h5
```
The generated prediction are stored in `data/{metr-la|pems-bay}/results/`.

## Graph Construction
DCRNN requires pre-calculated road network distances, Graph Wavenet allows to compute them implicitly.
The LSTM and the Transformer-based GMAN do not use road network distances.

The pairwise pre-calculated road network distances between sensors `data/{metr-la|pems-bay}/distances.csv` are used to generate the adjacency matrix:
```bash
python -m scripts.gen_adj_mx  --sensor_locations_filename=data/metr-la/graph_sensor_locations.csv --normalized_k=0.1\
    --output_pkl_filename=data/metr-la/adj_mx.pkl
```
The world-coordinate locations of the sensors are available at `data/{metr-la|pems-bay}/graph_sensor_locations.csv`.


## Model Training & Evaluation
To be added.

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following papers:
```
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}

@article{wu_graph_2019,
  title = {Graph {WaveNet} for Deep Spatial-Temporal Graph Modeling},
  url = {http://arxiv.org/abs/1906.00121},
  author = {Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  year = {2019},
}

@inproceedings{GMAN-AAAI2020,
  author = {Chuanpan Zheng and Xiaoliang Fan and Cheng Wang and Jianzhong Qi}
  title = {GMAN: A Graph Multi-Attention Network for Traffic Prediction},
  booktitle = {AAAI},
  year = {2020}
}
```
