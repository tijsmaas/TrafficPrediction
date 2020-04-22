import argparse
import numpy as np
import os
import pandas as pd

from scripts.generate_training_data import generate_graph_seq2seq_io_data

def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    sh = x.shape
    print("Dataframe contains", sh[-1], "fields,",sh[2],"for nodes,")
    print(sh[1],"timeframes per hour,",sh[0],"hours (approx.", sh[0]//(24*7),"weeks)")
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # x shape:  (34249, 12, 207, 2) , y shape:  (34249, 12, 207, 2)
    print(x_offsets)
    print(x[0,0,0,:])
    print(x[0,1,0,:])
    print(x[0,2,0,:])

    


def main(args):
    print("Inspecting", args.traffic_df_filename)
    generate_train_val_test(args)


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
    main(args)
