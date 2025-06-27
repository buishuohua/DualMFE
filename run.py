from scripts import train, test
import argparse
import os
import sys
from utils.exceptions import ModeNotSupported


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="train",
                        choices=["train", "test"], required=True, help="Mode to run: train or test")

    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("-mod", "--model", type=str, default="GRU",
                        choices=["GRU", "vTransformer", "iTransformer"], required=True, help="Model to use: GRU")
    parser.add_argument("-n", "--n_heads", type=int, default=8,
                        help="Number of heads for transformer")
    parser.add_argument("-d", "--d_model", type=int, default=256,
                        help="Dimension of model")
    parser.add_argument("-ff", "--d_ff", type=int, default=1024,
                        help="Dimension of feedforward layer")
    parser.add_argument("-nb", "--n_blocks", type=int, default=8,
                        help="Number of blocks for transformer")
    parser.add_argument("-a", "--activation", type=str, default="gelu",
                        choices=["gelu", "relu", "elu"], help="Activation function")
    parser.add_argument("-do", "--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("-add", "--attn_dropout", type=float, default=0.2,
                        help="Dropout rate for attention")
    parser.add_argument("-ms", "--mask_flag", default=False, action="store_true",
                        help="Whether to use mask flag for attention")


    parser.add_argument("-cl", "--continue_learning", default=False, action="store_true",
                        help="Continue learning: Set to true to continue learning, need to specify the experiment name, if not, starting from the latest experiment")
    parser.add_argument("-e", "--expr", type=str, default=None,
                        help="Experiment name need to continue learning or test")

    parser.add_argument('-w', "--lookback_window", type=int, default=120,
                        help="Lookback window size")
    parser.add_argument("-st", "--stride", type=int, default=120,
                        help="Downsampling stride")
    parser.add_argument("-fe", "--feature_engineering", default=False, action="store_true",
                        help="Whether to do feature engineering")

    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument('-lr', "--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument('-o', "--optimizer", type=str, default="AdamW",
                        choices=["Adam", "AdamW"], help="Optimizer to use")

    return parser.parse_args()

def run():
    args = parse_args()
    proj_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(proj_root)
    args.root_path = proj_root
    if args.mode == "train":
        train.main(args)
    elif args.mode == "test":
        test.main(args)
    else:
        raise ModeNotSupported(f"{args.mode} not supported")

if __name__ == "__main__":
    run()
