import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description=os.path.basename(os.path.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="resnet50", help="see network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=100, help="input batch size")
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--parallel-val", type=bool, default=True, help="use a separate thread for validation")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="cifar", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=32, help="length of hashing binary")

    # special settings
    parser.add_argument("--lr-hc", type=float, default=1e-4, help="learning rate of hash centers")
    parser.add_argument("--type-of-distance", type=str, default="cosine", help="cosine/euclidean/squared_euclidean")
    parser.add_argument("--type-of-triplets", type=str, default="all", help="all/semi-hard/hard")
    parser.add_argument("--margin", type=float, default=0.25, help="margin for the triplet loss")

    parser.add_argument("--lambda1", type=float, default=1, help="hyper-parameter for loss TL")
    parser.add_argument("--lambda2", type=float, default=0, help="hyper-parameter for loss CL")
    parser.add_argument("--lambda3", type=float, default=1, help="hyper-parameter for loss XL")
    parser.add_argument("--lambda4", type=float, default=1, help="hyper-parameter for loss CTL")

    parser.add_argument("--warmup", type=int, default=15, help="when to calc CTL")

    parser.add_argument("--lr-ma", type=float, default=1, help="lr for moving average strategy")

    args = parser.parse_args()

    # mods
    # args.type_of_triplets = "hard"
    # args.lr_ma = 1e-5
    # args.batch_size = 128
    # args.warmup = 10

    return args
