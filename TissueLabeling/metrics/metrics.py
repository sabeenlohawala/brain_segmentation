import lightning as L
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torchmetrics import Metric
import wandb

from TissueLabeling.brain_utils import mapping

class Dice(Metric):
    def __init__(self, fabric, config, **kwargs):
        super().__init__(**kwargs)
        self.add_state("class_dice", default=torch.zeros((1,51)), dist_reduce_fx="mean") # want to average across all gpus
        self.add_state("dice", default=torch.Tensor((1,)), dist_reduce_fx="mean") # want to average across all gpus

        # weights for weighted dice score
        pixel_counts = torch.from_numpy(np.load(f"{config.data_dir}/pixel_counts.npy"))

        if config.nr_of_classes == 2:
            pixel_counts = torch.from_numpy(np.array([pixel_counts[0],sum(pixel_counts[1:])])) # uncomment for binary classification
        elif config.nr_of_classes == 7:
            new_indices = mapping(torch.tensor(list(range(107))),nr_of_classes=config.nr_of_classes,original=False)
            unique_indices = np.unique(new_indices)
            new_counts = torch.zeros(config.nr_of_classes)
            for ind in unique_indices:
                mask = (new_indices == ind)
                new_counts[ind] = torch.sum(pixel_counts[mask])
            pixel_counts = new_counts
            
        self.smooth = 1e-7
        self.weights = 1 / (pixel_counts + self.smooth)
        self.weights = self.weights / self.weights.sum()
        self.weights = fabric.to_device(self.weights)
        self.nr_of_classes = config.nr_of_classes

    def update(self, target: Tensor, preds: Tensor) -> None:
        '''
        This function updates the state variables specified in __init__ based on the target and the predictions
        as suggested here: https://arxiv.org/abs/2004.10664

        Args:
            target (torch.tensor): Ground-truth mask. Tensor with shape [B, 1, H, W]
            preds (torch.tensor): Predicted class probabilities. Tensor with shape [B, C, H, W]
        '''

        # convert mask to one-hot
        y_true_oh = torch.nn.functional.one_hot(
            target.long().squeeze(1), num_classes=self.nr_of_classes
        ).permute(0, 3, 1, 2)

        # class specific intersection and union: sum over voxels
        class_intersect = torch.sum((self.weights.view(1, -1, 1, 1) * (y_true_oh * preds)), axis=(2, 3))
        class_union = torch.sum((self.weights.view(1, -1, 1, 1) * (y_true_oh + preds)), axis=(2, 3))

        # overall intersection and union: sum over classes
        intersect = torch.sum(class_intersect,axis=1)
        union = torch.sum(class_union,axis=1)

        # average over samples in the batch
        self.class_dice = torch.mean(2.0 * class_intersect / (class_union + self.smooth), axis=0)
        self.dice = torch.mean(2.0 * intersect / (union + self.smooth), axis=0)

    def compute(self) -> Tensor:
        '''
        This function computes the loss value based on the stored state.
        '''
        return 1 - self.dice, self.class_dice

class Classification_Metrics:
    def __init__(self, nr_of_classes: int, prefix: str, wandb_on: bool):
        # init
        self.nr_of_classes = nr_of_classes
        self.prefix = prefix
        self.wandb_on = wandb_on

        self.loss = []
        self.classDice = []

        self.Assert = torch.Tensor([1])

    def compute(
        self, y_true: torch.tensor, y_pred: torch.tensor, loss: float, classDice #, class_intersect, class_union
    ):
        self.loss.append(loss)
        if classDice is not None:
            self.classDice.append(classDice.tolist())

    def log(self, epoch, commit: bool = False, writer=None):
        logging_dict = {
            f"{self.prefix}/Loss": sum(self.loss) / len(self.loss),
            f"{self.prefix}/DICE/overall": sum([1 - item for item in self.loss])
            / len(self.loss),
            f"Assert": self.Assert.item(),
        }
        # for i in range(len(self.classDice[-1])):
        #     logging_dict[f"{self.prefix}/SegDICE/{i}"] = self.classDice[-1][i]

        if self.wandb_on:
            wandb.log(logging_dict, commit=commit)
        if writer is not None:
            writer.add_scalar(f"{self.prefix}/Loss", sum(self.loss) / len(self.loss),epoch)
            writer.add_scalar(
                f"{self.prefix}/DICE/overall",
                sum([1 - item for item in self.loss]) / len(self.loss), epoch,
            )

    def reset(self):
        # reset
        self.loss = []
        self.classDice = []
        self.Assert = torch.Tensor([1])

    def sync(self, fabric):
        self.Assert = fabric.all_reduce(self.Assert, reduce_op="sum")

        # self.Assert equals one for each process. The sum must thus be equal to the number of processes in ddp strategy
        # assert self.Assert.item() == torch.cuda.device_count(), f"Metrics Syncronization Did not Work. Assert: {self.Assert}, Devices {torch.cuda.device_count()}"
