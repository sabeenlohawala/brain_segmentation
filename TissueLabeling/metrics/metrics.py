import lightning as L
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torchmetrics import Metric
import wandb

from TissueLabeling.brain_utils import mapping

class Dice_v2(Metric):
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

        self.smooth = 1e-7
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
        if self.nr_of_classes == 50:
            target[target >= 50] = 0

        # convert mask to one-hot
        y_true_oh = torch.nn.functional.one_hot(
            target.long().squeeze(1), num_classes=self.nr_of_classes
        ).permute(0, 3, 1, 2)

        weights = y_true_oh.sum(axis=(0,2,3))
        weights = 1 / (weights)
        # set inf weights (when count = 0) to max(weights)
        weights[weights == float('inf')] = -float('inf')
        weights[weights == -float('inf')] = torch.max(weights)
        weights = weights / weights.sum()

        class_intersect = torch.sum(y_true_oh * preds, axis=(2,3)) # [batch_size, nr_of_classes]
        class_union = torch.sum(y_true_oh + preds, axis=(2,3)) # [batch_size, nr_of_classes]

        # sum over classes
        overall_intersect = (class_intersect * weights.view(1,-1)).sum(axis=1)
        overall_union = (class_union * weights.view(1,-1)).sum(axis=1)

        # average over batch
        self.class_dice = 2.0 * (torch.mean(class_intersect, axis=0) + self.smooth) / (torch.mean(class_union, axis=0) + self.smooth)
        self.class_dice[self.class_dice > 1] = 0
        self.dice = torch.mean(2.0 * (overall_intersect + self.smooth) / (overall_union + self.smooth), axis=0)

    def compute(self) -> Tensor:
        """
        This function computes the loss value based on the stored state.
        """
        score = 1 - self.dice if self.is_loss else self.dice
        if self.class_specific_scores:
            return score, self.class_dice
        return score

class Dice(Metric):
    """
    Compute the multi-class generlized dice loss as suggested here:
    https://arxiv.org/abs/2004.10664
    """
    def __init__(self, fabric, config, is_loss=False, class_specific_scores=False, **kwargs):
        """
        Constructor.

        Args:
            fabric (L.Fabric | None): fabric to get right device
            config (TissueLabeling.config.Configuration): contains the experiment parameters
            is_loss (bool): whether to return the dice loss = 1 - dice
            class_specific_scores (bool): whether to return class_dice
        """
        super().__init__(**kwargs)
        self.add_state("class_dice", default=torch.zeros((1,config.nr_of_classes)), dist_reduce_fx="mean") # want to average across all gpus
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
        elif config.nr_of_classes == 50:
            pixel_counts[0] += pixel_counts[50]
            pixel_counts = pixel_counts[:-1]
        
        self.smooth = 1e-7
        self.weights = 1 / (pixel_counts + self.smooth)
        self.weights = self.weights / self.weights.sum()
        if fabric is not None:
            self.weights = fabric.to_device(self.weights)
        self.nr_of_classes = config.nr_of_classes

        self.is_loss = is_loss
        self.class_specific_scores = class_specific_scores

    def update(self, target: Tensor, preds: Tensor) -> None:
        '''
        This function updates the state variables specified in __init__ based on the target and the predictions
        as suggested here: https://arxiv.org/abs/2004.10664

        Args:
            target (torch.tensor): Ground-truth mask. Tensor with shape [B, 1, H, W]
            preds (torch.tensor): Predicted class probabilities. Tensor with shape [B, C, H, W]
        '''
        if self.nr_of_classes == 50:
            target[target >= 50] = 0

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
        score = 1 - self.dice if self.is_loss else self.dice
        if self.class_specific_scores:
            return score, self.class_dice
        return score

class Classification_Metrics:
    """
    This class is used to accumulate and log the loss and metric across a batch of samples.
    """
    def __init__(self, nr_of_classes: int, prefix: str, wandb_on: bool, loss_name = "Dice", metric_name = "Dice"):
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
        self.class_dice = torch.zeros((self.nr_of_classes,),device='cpu')

        self.Assert = torch.Tensor([1])

    def compute(self, loss: float, metric: float, class_dice=None): #, class_intersect, class_union):
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
            f"{self.prefix}/Loss/{self.loss_name.title()}": sum(self.loss) / len(self.loss),
            f"{self.prefix}/Metric/{self.metric_name.title()}": sum(self.metric) / len(self.metric),
            f"Assert": self.Assert.item(),
        }
        
        if len(self.class_dice) > 0:
            for i in range(len(self.class_dice)):
                logging_dict[f"{self.prefix}/Metric/ClassDice/{i}"] = self.class_dice[i].item() / len(self.loss)

        if self.wandb_on:
            wandb.log(logging_dict, commit=commit)
        if writer is not None:
            # writer.add_scalar(f"{self.prefix}/Loss/{self.loss_name.title()}", sum(self.loss) / len(self.loss), epoch)
            # writer.add_scalar(f"{self.prefix}/Metric/{self.metric_name.title()}", sum(self.metric) / len(self.metric), epoch)
            for key, val in logging_dict.items():
                if key != 'Assert':
                    writer.add_scalar(key, val, epoch)

    def reset(self):
        # reset
        self.loss = []
        self.metric = []
        self.class_dice = torch.zeros((self.nr_of_classes,),device='cpu')
        self.Assert = torch.Tensor([1])

    def sync(self, fabric):
        self.Assert = fabric.all_reduce(self.Assert, reduce_op="sum")

        # self.Assert equals one for each process. The sum must thus be equal to the number of processes in ddp strategy
        # assert self.Assert.item() == torch.cuda.device_count(), f"Metrics Syncronization Did not Work. Assert: {self.Assert}, Devices {torch.cuda.device_count()}"
