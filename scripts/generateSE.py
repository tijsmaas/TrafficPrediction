import argparse
import os

import scripts
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

def write_edgelist(adj_file, edgelist_file):
    adj = np.load(adj_file, allow_pickle=True)[2]
    with open(edgelist_file, 'w') as f:
        n_nodes = adj.shape[0]
        for i in range(n_nodes):
            for j in range(n_nodes):
                w = adj[i][j]
                f.write(str(i) + ' ' + str(j) + ' ' + str(w) + '\n')

def read_graph(edgelist_file):
    G = nx.read_edgelist(
        edgelist_file, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())
    return G

def learn_embeddings(walks, dimensions, iter, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, size = dimensions, window = 10, min_count = 0, sg = 1,
        workers = 4, iter = iter)
    print ('Writing embedding to', output_file)
    model.wv.save_word2vec_format(output_file)

def main(args):
    # Author settings
    is_directed = True
    dimensions = 64
    window_size = 10
    p = 2
    q = 1

    write_edgelist(args.adj_file, args.edgelist_file)
    nx_G = read_graph(args.edgelist_file)
    G = scripts.Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, dimensions, args.iter, args.SE_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adj_file", type=str, default="data/sensor_graph/adj_mx.pkl", help="Input file adjacency matrix to build graph."
    )
    parser.add_argument(
        "--SE_file", type=str, help="Output file with sensor embeddings. (e.g. data/sensor_graph/SE.txt)",
    )
    parser.add_argument('--walk_length', type=int, default=80,
                        help='Length of random walks')
    parser.add_argument('--num_walks', type=int, default=100,
                        help='Number of random walks per iteration')
    parser.add_argument('--iter', type=int, default=1000,
                        help='Number of iterations')

    args = parser.parse_args()
    basepath = os.path.dirname(args.adj_file)
    args.edgelist_file = os.path.join(basepath, 'edgelist.txt')
    if not args.SE_file:
        args.SE_file = os.path.join(basepath, 'SE.txt')
    main(args)