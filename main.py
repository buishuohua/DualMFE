from scripts import train, test
import argparse
import os
import sys
from utils.exceptions import ModeNotSupported

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="train",
                        choices=["train", "test"], required=True, help="Mode to run: train or test")
    #TODO: Add more model choices
    parser.add_argument("-mod", "--model", type=str, default="GRU",
                        choices=["GRU", "vTransformer"], required=True, help="Model to use: GRU")

    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("-cl", "--continue_learning", default=False, action="store_true",
                        help="Continue learning: Set to true to continue learning, need to specify the experiment name, if not, starting from the latest experiment")

    parser.add_argument("-e", "--expr", type=str, default=None, help="Experiment name need to continue learning or test")
    parser.add_argument('-d', "--default")
    return parser.parse_args()

def main():
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
    main()
