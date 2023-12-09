import argparse
import os


def get_args():
    """
    This is the main function of the program.
    It parses command line arguments and executes the program logic.
    """

    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    subparsers = parser.add_subparsers(help="sub-command help")

    # create subparser for "resume-train" command
    resume = subparsers.add_parser(
        "resume-train", help="Use this sub-command for resuming training"
    )
    resume.add_argument(
        "logdir",
        type=str,
        help="Folder containing previous checkpoints",
    )
    resume.add_argument("--debug", action="store_true", dest="debug")

    # create subparser for "train" command
    train = subparsers.add_parser("train", help="Use this sub-command for training")

    # Add command line arguments
    train.add_argument(
        "--logdir",
        help="Tensorboard directory",
        type=str,
        required=False,
        default=os.getcwd(),
    )
    train.add_argument(
        "--model_name",
        help="Name of model to use for segmentation",
        type=str,
        default="segformer",
    )
    train.add_argument(
        "--num_epochs",
        help="Number of epochs to train",
        type=int,
        required=False,
        default=20,
    )
    train.add_argument(
        "--batch_size",
        help="Batch size for training",
        type=int,
        required=False,
        default=64,
    )
    train.add_argument(
        "--lr", help="Learning for training", type=float, required=False, default=6e-5
    )
    train.add_argument(
        "--data_dir",
        help="Directory of which dataset to train on",
        type=str,
        default="/om2/user/sabeen/nobrainer_data_norm/new_small_no_aug_51",
    )
    train.add_argument(
        "--pretrained",
        help="Whether to use pretrained model",
        type=bool,
        required=False,
        default=True,
    )
    train.add_argument(
        "--seed", help="Random seed value", type=int, required=False, default=42
    )

    # Parse the command line arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print(get_args())
