
import torch
import wandb
from torch import nn
import numpy as np
import lightning as L

DATA_USER = 'sabeen' # alternatively, 'matth406'
# DATASET = 'medium'

class Dice(nn.Module):

    def __init__(self, nr_of_classes: int, fabric: L.Fabric, smooth: float = 1e-7) -> None:
        super(Dice, self).__init__()
        '''
        Compute the multi-class generlized dice loss as suggested here:
        https://arxiv.org/abs/2004.10664

        Args:
            nr_of_classes (int): number of classes
            fabric (L.Fabric): fabric to get right device
            smooth (float, optional): Smoothing Constant. Defaults to 1e-7.
        '''
        
        # init
        self.nr_of_classes = nr_of_classes
        self.smooth = smooth

        pixel_counts = np.load(f'/om2/user/{DATA_USER}/nobrainer_data_norm/new_small_no_aug_51/pixel_counts.npy')
        # NR_OF_CLASSES = 6
        # pixel_counts = np.load(f'/om2/user/sabeen/nobrainer_data_norm/matth406_medium_6_classes/pixel_counts.npy')
        pixel_counts = torch.from_numpy(pixel_counts)

        # pixel_counts = torch.from_numpy(np.array([pixel_counts[0],sum(pixel_counts[1:])])) # uncomment for binary classification

        self.weights = 1/pixel_counts
        # Check for inf values
        inf_mask = torch.isinf(self.weights)
        # Replace inf values with zero
        self.weights[inf_mask] = 0.0
        # normalize weights
        self.weights = self.weights/self.weights.sum()

        # send weights to GPU        
        self.weights = fabric.to_device(self.weights)
        self.denom = None

    def forward(self, y_true: torch.tensor, y_pred: torch.tensor):
        '''
        Args:
            y_true (torch.tensor): Ground Truth class. Tensor of shape [B,1,H,W]
            y_pred (torch.tensor): Predicted class probabilities. Tensor of shape [B,C,H,W]

        Returns:
            float: differentiable dice loss
        '''
        # one-hot encode label tensor
        y_true_oh = torch.nn.functional.one_hot(y_true.squeeze(1), num_classes=self.nr_of_classes).permute(0,3,1,2)
        
        # compute the generalized dice for each imamge
        class_intersect = torch.sum(self.weights.view(1,-1,1,1)*(y_true_oh * y_pred), axis=(2,3))
        class_denom = torch.sum(self.weights.view(1,-1,1,1)*(y_true_oh + y_pred), axis=(2,3))

        intersect = torch.sum(class_intersect,axis=1)
        denom = torch.sum(class_denom,axis=1)

        classDice = torch.mean(2. * class_intersect / (class_denom + self.smooth),axis=0)
        
        # compute the average over the batch
        dice_coeff = torch.mean((2. * intersect / (denom + self.smooth)))
        dice_loss = 1 - dice_coeff
        
        return dice_loss, classDice

class Classification_Metrics():

    def __init__(self, nr_of_classes: int, prefix: str, wandb_on: bool):

        # init
        self.nr_of_classes = nr_of_classes
        self.prefix = prefix
        self.wandb_on = wandb_on

        self.loss = []
        self.classDice = []
        # self.TP, self.TN, self.FP, self.FN = torch.zeros(nr_of_classes),torch.zeros(nr_of_classes),torch.zeros(nr_of_classes),torch.zeros(nr_of_classes)

        self.Assert = torch.Tensor([1])

    def compute(self, y_true: torch.tensor, y_pred: torch.tensor, loss: float, classDice):

        self.loss.append(loss)
        self.classDice.append(classDice.tolist())

        # y_pred_hard = y_pred.argmax(1, keepdim=True)
        # for i in range(self.nr_of_classes):
        #     # TP
        #     self.TP[i] += (y_pred_hard[y_true == i] == i).sum().item()
        #     # TN
        #     self.TN[i] += (y_pred_hard[y_true != i] != i).sum().item()
        #     # FP
        #     self.FP[i] += (y_pred_hard[y_true == i] != i).sum().item()
        #     # FN
        #     self.FN[i] += (y_pred_hard[y_true != i] == i).sum().item()

    # def nans_to_zero(self, array: torch.tensor):
        
    #     where_are_NaNs = torch.isnan(array)
    #     array[where_are_NaNs] = 0

    #     return array

    # def accuracy(self):

    #     accruacy_per_class = (self.TP + self.TN)/(self.TP + self.TN + self.FP+self.FN)
    #     macro_accuracy = self.nans_to_zero(accruacy_per_class).mean()

    #     return macro_accuracy
    
    # def f1(self):

    #     f1_per_class = (2*self.TP)/(2*self.TP+self.FN+self.FP)
    #     macro_f1 = self.nans_to_zero(f1_per_class).mean()

    #     return macro_f1
    
    # def recall(self):

    #     recall_per_class = self.TP / (self.TP+self.FN)
    #     macro_recall = self.nans_to_zero(recall_per_class).mean()

    #     return macro_recall

    # def precision(self):

    #     precision_per_class = self.TP / (self.TP+self.FP)
    #     macro_precision = self.nans_to_zero(precision_per_class).mean()

    #     return macro_precision
    
    def log(self, commit: bool = False, writer=None):
        logging_dict = {
            f"{self.prefix}/Loss": sum(self.loss)/len(self.loss),
            f"{self.prefix}/DICE/overall": sum([1 - item for item in self.loss])/len(self.loss),
            # f"{self.prefix}/Accuracy": self.accuracy().item(),
            # f"{self.prefix}/F1": self.f1().item(),
            # f"{self.prefix}/Recall": self.recall().item(),
            # f"{self.prefix}/Precision": self.precision().item(),
            f"Assert": self.Assert.item(),
        }

        # for i in range(len(self.classDice[-1])):
        #     logging_dict[f"{self.prefix}/SegDICE/{i}"] = self.classDice[-1][i]
        if self.wandb_on:
            wandb.log(logging_dict, commit=commit)
        if writer is not None:
            writer.add_scalar(f"{self.prefix}/Loss", sum(self.loss)/len(self.loss))
            writer.add_scalar(f"{self.prefix}/DICE/overall", sum([1 - item for item in self.loss])/len(self.loss))

    def reset(self):

        # reset
        self.loss = []
        # self.TP, self.TN, self.FP, self.FN = torch.zeros(self.nr_of_classes),torch.zeros(self.nr_of_classes),torch.zeros(self.nr_of_classes),torch.zeros(self.nr_of_classes)
        self.Assert = torch.Tensor([1])

    def sync(self, fabric):
        # self.TP = fabric.all_reduce(self.TP, reduce_op="sum")
        # self.TN = fabric.all_reduce(self.TN, reduce_op="sum")
        # self.FP = fabric.all_reduce(self.FP, reduce_op="sum")
        # self.FN = fabric.all_reduce(self.FN, reduce_op="sum")
        self.Assert = fabric.all_reduce(self.Assert, reduce_op="sum")

        # self.Assert equals one for each process. The sum must thus be equal to the number of processes in ddp strategy
        # assert self.Assert.item() == torch.cuda.device_count(), f"Metrics Syncronization Did not Work. Assert: {self.Assert}, Devices {torch.cuda.device_count()}"