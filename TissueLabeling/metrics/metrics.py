import lightning as L
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torchmetrics import Metric
import wandb

class Dice(Metric):
    def __init__(self, fabric, config, **kwargs):
        super().__init__(**kwargs)
        self.add_state("class_dice", default=torch.zeros((1,51)), dist_reduce_fx="mean") # want to average across all gpus
        self.add_state("dice", default=torch.Tensor((1,)), dist_reduce_fx="mean") # want to average across all gpus

        # weights for weighted dice score
        pixel_counts = torch.from_numpy(np.load(f"{config.data_dir}/pixel_counts.npy"))
        self.weights = 1 / pixel_counts
        self.weights = self.weights / self.weights.sum()
        self.weights = fabric.to_device(self.weights)
        self.smooth = 1e-7
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

class OldDice(nn.Module):
    def __init__(
        self, fabric: L.Fabric, config, smooth: float = 1e-7
    ) -> None:
        """
        Compute the multi-class generlized dice loss as suggested here:
        https://arxiv.org/abs/2004.10664

        Args:
            nr_of_classes (int): number of classes
            fabric (L.Fabric): fabric to get right device
            smooth (float, optional): Smoothing Constant. Defaults to 1e-7.
        """
        super(Dice, self).__init__()

        # init
        self.nr_of_classes = config.nr_of_classes
        self.smooth = smooth

        pixel_counts = np.load(f"{config.data_dir}/pixel_counts.npy")

        # NR_OF_CLASSES = 6
        # pixel_counts = np.load(f'/om2/user/sabeen/nobrainer_data_norm/matth406_medium_6_classes/pixel_counts.npy')
        pixel_counts = torch.from_numpy(pixel_counts)

        # pixel_counts = torch.from_numpy(np.array([pixel_counts[0],sum(pixel_counts[1:])])) # uncomment for binary classification

        self.weights = 1 / pixel_counts
        # Check for inf values
        inf_mask = torch.isinf(self.weights)
        # Replace inf values with zero
        self.weights[inf_mask] = 0.0
        # normalize weights
        self.weights = self.weights / self.weights.sum()

        # send weights to GPU
        self.weights = fabric.to_device(self.weights)
        self.denom = None

    def forward(self, y_true: torch.tensor, y_pred: torch.tensor, debug=False):
        """
        Args:
            y_true (torch.tensor): Ground Truth class. Tensor of shape [B,1,H,W]
            y_pred (torch.tensor): Predicted class probabilities. Tensor of shape [B,C,H,W]

        Returns:
            float: differentiable dice loss
        """
        # one-hot encode label tensor
        y_true_oh = torch.nn.functional.one_hot(
            y_true.squeeze(1), num_classes=self.nr_of_classes
        ).permute(0, 3, 1, 2)

        # compute the generalized dice for each imamge
        self.class_intersect = torch.sum(
            self.weights.view(1, -1, 1, 1) * (y_true_oh * y_pred), axis=(2, 3)
        )
        self.class_union = torch.sum(
            self.weights.view(1, -1, 1, 1) * (y_true_oh + y_pred), axis=(2, 3)
        )

        intersect = torch.sum(self.class_intersect, axis=1)
        denom = torch.sum(self.class_union, axis=1)

        classDice = torch.mean(
            2.0 * self.class_intersect / (self.class_union + self.smooth), axis=0
        )

        # compute the average over the batch
        dice_coeff = torch.mean((2.0 * intersect / (denom + self.smooth)))
        dice_loss = 1 - dice_coeff

        return dice_loss, classDice


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
