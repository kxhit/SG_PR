import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run SimGNN.")

    parser.add_argument("--graph_pairs_dir", nargs = "?",
                        default = "/media/work/data/kitti/odometry/semantic-kitti/5_20/graph_pairs_sem_5_20_ds/",
                        help = "Folder with graph pair jsons.")
    parser.add_argument("--p_thresh", type=int, default=3,
                        help="Threshold for positive samples. Default is 3m.")
    parser.add_argument("--fold", type=str, default="00",
                        help="fold xx to eval. Default is 00.")
    parser.add_argument("--epochs", type = int, default = 500,
                        help = "Number of training epochs. Default is 500.")
    parser.add_argument("--logdir",
                        type=str,
                        default="./logs",
                        help="Output log directory. Default is ./logs.")
    parser.add_argument("--model",
                        type=str,
                        default="",
                        help="model.")
    parser.add_argument("--keep-node",
                        type = float,
                        default=1,
                        help="Randomly delete some nodes and edges. Default is 1")
    parser.add_argument("--filters-1", type = int, default = 64,
                        help = "Filters (neurons) in 1st convolution. Default is 64.")
    parser.add_argument("--filters-2", type = int, default = 64,
                        help = "Filters (neurons) in 2nd convolution. Default is 64.")
    parser.add_argument("--filters-3", type = int, default = 32,
                        help = "Filters (neurons) in 3rd convolution. Default is 32.")
    parser.add_argument("--tensor-neurons", type = int, default = 16,
                        help = "Neurons in tensor network layer. Default is 16.")
    parser.add_argument("--bottle-neck-neurons", type = int, default = 16,
                        help = "Bottle neck layer neurons. Default is 16.")
    parser.add_argument("--batch-size", type = int, default = 128,
                        help = "Number of graph pairs per batch. Default is 128.")
    parser.add_argument("--dropout", type = float, default = 0,
                        help = "Dropout probability. Default is 0.")
    parser.add_argument("--learning-rate", type = float, default = 0.001,
                        help = "Learning rate. Default is 0.001.")
    parser.add_argument("--weight-decay", type = float, default = 5*10**-4,
                        help = "Adam weight decay. Default is 5*10^-4.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id to use. Default is 0.")
    
 
    return parser.parse_args()
