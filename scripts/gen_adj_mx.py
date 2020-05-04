from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle

# Modified to use graph_sensor_locations.csv rather than the sensor_ids file.
# Modified to use coordinate distances instead of road distances

# This only computes approximate distance and does not take into account landscape features (e.g. heights)
def distanceInKmBetweenEarthCoordinates(c1, c2):
    lat1, lon1 = c1
    lat2, lon2 = c2
    dLat = np.radians(lat2-lat1)
    dLon = np.radians(lon2-lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.sin(dLon/2) * np.sin(dLon/2) * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    earth_radius_km = 6371
    return earth_radius_km * c

def get_sensor_coords(sensor_locations_filename):
    # Prepare sensor coordinates (distance)
    sensor_ids_df = pd.read_csv(sensor_locations_filename, dtype={'index': 'int', 'sensor_id': 'str',
                                                                 'latitude': 'float', 'longitude': 'float'})
    sensor_coords = [None] * len(sensor_ids_df.values)
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for index, sensor_id, lat, long in sensor_ids_df.values:
        sensor_id_to_ind[sensor_id] = index
        sensor_coords[index] = np.array([lat, long])
    return sensor_id_to_ind, np.array(sensor_coords)

def get_road_distances(distances_filename, sensor_ids):
    # param distance_df: data frame with three columns: [from, to, distance].
    distance_df = pd.read_csv(distances_filename, dtype={'from': 'str', 'to': 'str'})
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    return sensor_id_to_ind, dist_mx


def get_adjacency_matrix(dist_mx, sensor_ids, normalized_k=0.1):
    """
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_locations_filename', type=str, default='data/sensor_graph/graph_sensor_locations.csv',
                        help='File containing sensor locations, to obtain the right sensor id ordering.')
    parser.add_argument('--sensor_ids_filename', type=str,  # default='data/sensor_graph/graph_sensor_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str,  # default='data/sensor_graph/distances_la_2012.csv',
                        help='CSV file containing sensor travel distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='data/sensor_graph/adj_mx.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    if not args.sensor_ids_filename or not args.distances_filename:
        sensor_id_to_ind, sensor_coords = get_sensor_coords(args.sensor_locations_filename)
        dist_km = []
        for idx, c1 in enumerate(sensor_coords):
            dist_km.append([distanceInKmBetweenEarthCoordinates(c1, c2) for c2 in sensor_coords])
        dist_mx = np.array(dist_km)

    if args.sensor_ids_filename:
        with open(args.sensor_ids_filename) as f:
            sensor_ids = f.read().strip().split(',')
    else:
        sensor_ids = list(sensor_id_to_ind.keys())

    if args.distances_filename:
        sensor_id_to_ind, dist_mx = get_road_distances(args.distances_filename, sensor_ids)

    adj_mx = get_adjacency_matrix(dist_mx, sensor_ids)
    # Fix formatting
    adj_mx = adj_mx.astype(np.float32)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
