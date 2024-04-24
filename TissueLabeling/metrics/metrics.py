import os

import lightning as L
import numpy as np
import pickle
import torch
from torch import nn
from torch import Tensor
from torchmetrics import Metric
import wandb

from TissueLabeling.brain_utils import mapping


class Dice(Metric):
    """
    Compute the multi-class generlized dice loss as suggested here:
    https://arxiv.org/abs/2004.10664
    """

    def __init__(
        self, fabric, config, is_loss=False, class_specific_scores=False, **kwargs
    ):
        """
        Constructor.

        Args:
            fabric (L.Fabric | None): fabric to get right device
            config (TissueLabeling.config.Configuration): contains the experiment parameters
            is_loss (bool): whether to return the dice loss = 1 - dice
            class_specific_scores (bool): whether to return class_dice
        """
        super().__init__(**kwargs)
        self.add_state(
            "class_dice",
            default=torch.zeros((1, config.nr_of_classes)),
            dist_reduce_fx="mean",
        )  # want to average across all gpus
        self.add_state(
            "dice", default=torch.Tensor((1,)), dist_reduce_fx="mean"
        )  # want to average across all gpus

        # weights for weighted dice score
        # if config.new_kwyk_data:
        #     with open(
        #         os.path.join(config.data_dir, "train_pixel_counts.pkl"), "rb"
        #     ) as f:
        #         train_pixel_counts = pickle.load(f)

        #     # map counts from original freesurfer labels to nr_of_classes labels
        #     key_arr = np.array(list(train_pixel_counts.keys()))
        #     # mapped_keys = mapping(
        #     #     key_arr.copy(), nr_of_classes=config.nr_of_classes, original=True
        #     # ) # mapping mod
        #     mapped_keys = mapping(key_arr.copy(), config.nr_of_classes)
        #     mapped_pixel_counts = {
        #         class_num: sum(
        #             [
        #                 train_pixel_counts[i]
        #                 for i in key_arr[np.where(mapped_keys == class_num)]
        #             ]
        #         )
        #         if class_num in mapped_keys
        #         else 0
        #         for class_num in range(config.nr_of_classes)
        #     }

        #     # sort pixel count dict by key so numpy array indexing corresponds to correct label and counts
        #     sorted_counts = sorted(
        #         list((label, count) for label, count in mapped_pixel_counts.items()),
        #         key=lambda x: x[0],
        #     )
        #     pixel_counts = torch.tensor(list(count for _, count in sorted_counts))
        # else: # these prob don't work anymore
        #     pixel_counts = torch.from_numpy(
        #         np.load(f"{config.data_dir}/pixel_counts.npy")
        #     )

        #     if config.nr_of_classes == 2:
        #         pixel_counts = torch.from_numpy(
        #             np.array([pixel_counts[0], sum(pixel_counts[1:])])
        #         )  # uncomment for binary classification
        #     elif config.nr_of_classes == 7 or config.nr_of_classes == 17:
        #         new_indices = mapping(
        #             torch.tensor(list(range(51))),
        #             nr_of_classes=config.nr_of_classes,
        #             reference_col="50-class",
        #         )
        #         unique_indices = np.unique(new_indices)
        #         new_counts = torch.zeros(config.nr_of_classes)
        #         for ind in unique_indices:
        #             mask = new_indices == ind
        #             new_counts[ind] = torch.sum(pixel_counts[mask])
        #         pixel_counts = new_counts

        self.smooth = 1e-7
        # self.weights = 1 / (pixel_counts + self.smooth)
        # self.weights = self.weights / self.weights.sum()
        self.weights = torch.zeros((config.nr_of_classes,))
        if fabric is not None:
            self.weights = fabric.to_device(self.weights)
        self.nr_of_classes = config.nr_of_classes

        self.is_loss = is_loss
        self.class_specific_scores = class_specific_scores

    def update(self, target: Tensor, preds: Tensor) -> None:
        """
        This function updates the state variables specified in __init__ based on the target and the predictions
        as suggested here: https://arxiv.org/abs/2004.10664

        Args:
            target (torch.tensor): Ground-truth mask. Tensor with shape [B, 1, H, W]
            preds (torch.tensor): Predicted class probabilities. Tensor with shape [B, C, H, W]
        """

        unique, counts = torch.unique(target, return_counts=True)
        self.weights[:] = self.smooth
        for i,label in enumerate(unique):
            self.weights[label] += counts[i]
        self.weights = 1 / self.weights
        self.weights = self.weights / self.weights.sum()

        # convert mask to one-hot
        y_true_oh = torch.nn.functional.one_hot(
            target.long().squeeze(1), num_classes=self.nr_of_classes
        ).permute(0, 3, 1, 2)

        # class specific intersection and union: sum over voxels
        class_intersect = torch.sum(
            (self.weights.view(1, -1, 1, 1) * (y_true_oh * preds)), axis=(2, 3)
        )
        class_union = torch.sum(
            (self.weights.view(1, -1, 1, 1) * (y_true_oh + preds)), axis=(2, 3)
        )

        # overall intersection and union: sum over classes
        intersect = torch.sum(class_intersect, axis=1)
        union = torch.sum(class_union, axis=1)

        # average over samples in the batch
        self.class_dice = torch.mean(
            2.0 * class_intersect / (class_union + self.smooth), axis=0
        )
        self.dice = torch.mean(2.0 * intersect / (union + self.smooth), axis=0)

    def compute(self) -> Tensor:
        """
        This function computes the loss value based on the stored state.
        """
        score = 1 - self.dice if self.is_loss else self.dice
        if self.class_specific_scores:
            return score, self.class_dice
        return score


class Classification_Metrics:
    """
    This class is used to accumulate and log the loss and metric across a batch of samples.
    """

    def __init__(
        self,
        nr_of_classes: int,
        prefix: str,
        wandb_on: bool,
        loss_name="Dice",
        metric_name="Dice",
        class_specific_scores=False,
    ):
        """
        Constructor.

        Args:
            nr_of_classes (int): the number of classes to segment
            prefix (int): 'Train' or 'Validation'?
            wandb_on (bool): whether to log to wandb
            loss_name (str): 'dice' or 'focal'
            metric_name (str): 'dice'
        """
        self.nr_of_classes = nr_of_classes
        self.prefix = prefix
        self.wandb_on = wandb_on
        self.loss_name = loss_name
        self.metric_name = metric_name

        self.loss = []
        self.metric = []
        self.class_dice = torch.zeros((self.nr_of_classes,), device="cpu")

        self.Assert = torch.Tensor([1])
        self.class_specific_scores = class_specific_scores

    def compute(
        self, loss: float, metric: float, class_dice=None
    ):  # , class_intersect, class_union):
        """
        Appends the loss, metric, and class_dice to their respective lists so they can later
        be aggregated over the batch.
        """
        self.loss.append(loss)
        self.metric.append(metric)
        if class_dice is not None:
            self.class_dice = self.class_dice + class_dice.cpu()

    def log(self, epoch, commit: bool = False, writer=None):
        """
        Used to aggregate and log the stored losses and metrics for a batch to tensorboard and wandb.
        """
        logging_dict = {
            f"{self.prefix}/Loss/{self.loss_name.title()}": sum(self.loss)
            / len(self.loss),
            f"{self.prefix}/Metric/{self.metric_name.title()}": sum(self.metric)
            / len(self.metric),
            f"Assert": self.Assert.item(),
        }

        if len(self.class_dice) > 0 and self.class_specific_scores:
            for i in range(len(self.class_dice)):
                logging_dict[f"{self.prefix}/Metric/ClassDice/{i}"] = self.class_dice[
                    i
                ].item() / len(self.loss)

        if self.wandb_on:
            wandb.log(logging_dict, commit=commit)
        if writer is not None:
            # writer.add_scalar(f"{self.prefix}/Loss/{self.loss_name.title()}", sum(self.loss) / len(self.loss), epoch)
            # writer.add_scalar(f"{self.prefix}/Metric/{self.metric_name.title()}", sum(self.metric) / len(self.metric), epoch)
            for key, val in logging_dict.items():
                if key != "Assert":
                    writer.add_scalar(key, val, epoch)

    def reset(self):
        # reset
        self.loss = []
        self.metric = []
        self.class_dice = torch.zeros((self.nr_of_classes,), device="cpu")
        self.Assert = torch.Tensor([1])

    def sync(self, fabric):
        self.Assert = fabric.all_reduce(self.Assert, reduce_op="sum")

        # self.Assert equals one for each process. The sum must thus be equal to the number of processes in ddp strategy
        # assert self.Assert.item() == torch.cuda.device_count(), f"Metrics Syncronization Did not Work. Assert: {self.Assert}, Devices {torch.cuda.device_count()}"
