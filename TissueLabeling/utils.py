import csv
from datetime import datetime
import os
import random
import shutil
import numpy as np

import lightning as L
import torch
import wandb
from lightning.fabric import Fabric, seed_everything

def center_pad_tensor(input_tensor, new_height, new_width):
    # Get the dimensions of the input tensor
    _, height, width = input_tensor.size()

    # Calculate the amount of padding needed on each side
    pad_height = max(0, (new_height - height) // 2)
    pad_width = max(0, (new_width - width) // 2)

    # Calculate the total amount of padding needed
    pad_top = pad_height
    pad_bottom = new_height - height - pad_top
    pad_left = pad_width
    pad_right = new_width - width - pad_left

    # Apply padding
    padded_tensor = torch.nn.functional.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom))

    return padded_tensor

def main_timer(func):
    """Decorator to time any function"""

    def function_wrapper(*args,**kwargs):
        start_time = datetime.now()
        # print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        result = func(*args,**kwargs)

        end_time = datetime.now()
        # print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(
            f"Function: {func.__name__} Total runtime: {end_time - start_time} (HH:MM:SS)"
        )
        return result
    
    return function_wrapper


def set_seed(seed: int = 0) -> None:
    """Set the seed before GPU training

    Args:
        seed (int, optional): seed. Defaults to 0.
    """
    seed_everything(seed)

    if torch.cuda.is_available():
        # determines if cuda selects only deterministic algorithms or not
        # True = Only determinstic algo --> slower but reproducible
        torch.backends.cudnn.deterministic = False
        # determines if cuda should always select the same algorithms
        # (!! use only for fixed size inputs !!)
        # False = Always same algo --> slower but reproducible
        torch.backends.cudnn.benchmark = True


def init_cuda() -> None:
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        # Ampere GPUs (like A100) allow to use TF32 (which is faster than FP32)
        # see https://pytorch.org/docs/stable/notes/cuda.html
        # per default, TF32 is activated for convolutions
        print("Use TF32 for convolutions: ", torch.backends.cudnn.allow_tf32)
        # we manually activate it for matmul
        if "A100" in torch.cuda.get_device_name(0):
            torch.set_float32_matmul_precision("high")
        print("Use TF32 for matmul: ", torch.backends.cuda.matmul.allow_tf32)

        # reproducability vs speed (see set_seed function)
        # https://pytorch.org/docs/stable/notes/randomness.html
        print(
            "Only use determnisitc CUDA algorithms: ",
            torch.backends.cudnn.deterministic,
        )
        print(
            "Use the same CUDA algorithms for each forward pass: ",
            torch.backends.cudnn.benchmark,
        )


def init_wandb(
    project_name: str,
    fabric: L.fabric,
    model_params: dict,
    description: str,
) -> None:
    # check if staged artifacts exist:
    if os.path.exists(f"/home/{os.environ['USER']}/.local/share/wandb"):
        shutil.rmtree(f"/home/{os.environ['USER']}/.local/share/wandb")

    wandb.init(
        name=f"{fabric.device}-{datetime.now().month}-{datetime.now().day}-{datetime.now().hour}:{datetime.now().minute}",
        group=f"test-multigpu-{datetime.now().month}-{datetime.now().day}",
        # group=f'{datetime.now().month}-{datetime.now().day}-{datetime.now().hour}:{datetime.now().minute}',
        project=project_name,
        entity="tissue-labeling-sabeen",
        notes=description,
        config={**model_params},
        reinit=True,
        dir="/om2/scratch/Fri",
    )
    wandb.run.log_code(
        "./data",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    )
    wandb.run.log_code(
        "./models",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    )
    wandb.run.log_code(
        "./trainer",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    )


def init_fabric(**kwargs) -> L.fabric:
    fabric = Fabric(**kwargs)
    fabric.launch()

    if torch.cuda.device_count() > 1:
        # see: https://pytorch-lightning.readthedocs.io/en/1.9.0/_modules/lightning_fabric/strategies/ddp.html
        # fabric._strategy._ddp_kwargs['broadcast_buffers']=False

        # make environment infos available
        os.environ["RANK"] = str(fabric.global_rank)
        # local world size
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

        print(f"Initialize Process: {fabric.global_rank}")

    return fabric


def finish_wandb(out_file: str) -> None:
    """
    Finish Weights and Biases

    Args:
        out_file (str): name of the .out file of the run
    """

    # add .out file to wandb
    artifact_out = wandb.Artifact("OutFile", type="out_file")
    artifact_out.add_file(out_file)
    wandb.log_artifact(artifact_out)
    # finish wandb
    wandb.finish(quiet=True)
