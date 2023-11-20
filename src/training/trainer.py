
import torch
import webdataset as wds
import lightning as L
import wandb
from collections import Counter
from typing import Tuple

from training.logging import Log_Images
from models.metrics import Classification_Metrics

class Trainer():

    def __init__(
            self,
            model : torch.nn.Module,
            nr_of_classes: int,
            train_loader : wds.WebLoader,
            val_loader : wds.WebLoader,
            loss_fn : torch.nn.Module,
            optimizer : torch.optim.Optimizer,
            fabric : L.Fabric,
            save_every : int = 100000,
            ) -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.fabric = fabric
        self.save_every = save_every
        self.nr_of_classes = nr_of_classes

        if self.fabric.global_rank == 0:
            self.image_logger = Log_Images(self.fabric, nr_of_classes=nr_of_classes)

    def _save_state(self, epoch : int, batch_idx : int, log: bool = False) -> None:
        '''
        Save the pytorch model and the optimizer state

        Args:
            epoch (int): epoch number
            batch_idx (int): training batch index
            log (int): indicator if the model should be logged to wandb
        '''

        PATH = "/home/sabeen/brain_segmentation/models/checkpoint.ckpt"

        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model': self.model,
            'optimizer': self.optimizer,
            }
        
        self.fabric.save(PATH, state)

        if log:
            artifact = wandb.Artifact('Model', type='model')
            artifact.add_file(PATH)
            wandb.log_artifact(artifact)

    def optim_step(self, loss : torch.Tensor) -> None:
        '''
        One step of the optimizer

        Args:
            loss (torch.Tensor): computed loss for minibatch
        '''

        # faster than optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

        # optimizer step
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss is NaN or Inf")
        else:
            self.fabric.backward(loss)
            self.optimizer.step()

    def finish_wandb(self, out_file : str) -> None:
        '''
        Finish Weights and Biases

        Args:
            out_file (str): name of the .out file of the run
        '''

        # add .out file to wandb
        artifact_out = wandb.Artifact('OutFile', type='out_file')
        artifact_out.add_file(out_file)
        wandb.log_artifact(artifact_out)         
        # finish wandb
        wandb.finish(quiet=True)

    def train(self, epochs : int) -> None:
        
        batch_idx = 0

        self.train_metrics = Classification_Metrics(self.nr_of_classes, prefix="Train")
        self.validation_metrics = Classification_Metrics(self.nr_of_classes, prefix=f"Validation")

        print(f"Process {self.fabric.global_rank} starts training on {self.train_loader.length} batches per epoch over {epochs} epochs")

        for epoch in range(epochs):

            for image, mask, _ in self.train_loader:
                mask[mask != 0] = 1 # uncomment for binary classification check

                image = self.fabric.to_device(image)
                mask = self.fabric.to_device(mask)
                probs = self.__forward(image)
                self.__backward(probs, mask.long(), train=True)

                batch_idx += 1
                print(batch_idx)

            print(f"Process {self.fabric.global_rank} finished epoch {epoch}...")

            # evaluation

            print("start validation...")
            # compute loss on validation data
            self._validation(self.val_loader)
            # log loss and reset
            print(f"Process {self.fabric.global_rank} reached barrier")
            self.fabric.barrier()
            print(f"Process {self.fabric.global_rank} passed barrier")
            self.train_metrics.sync(self.fabric)
            self.validation_metrics.sync(self.fabric)
            print(f"Process {self.fabric.global_rank} synced tensors")

            if self.fabric.global_rank == 0:

                print("Rank:", self.fabric.global_rank)

                self.train_metrics.log(commit=False)
                self.validation_metrics.log(commit=False)

                print("saving image...")        
                self.image_logger.logging(self.model, epoch, commit=True)

            self.train_metrics.reset()
            self.validation_metrics.reset()

        if self.fabric.global_rank == 0:

            print("final saving model state...")
            self._save_state(epoch, batch_idx, log=True)

            # add .out file to wandb and terminate
            self.finish_wandb('/om2/user/sabeen/brain_segmentation/jobs/job_train.out')

    @torch.no_grad()
    def _validation(self, data_loader : wds.WebLoader) -> None:

        self.model.eval()
        for image, mask, _ in data_loader:
            mask[mask != 0] = 1 # uncomment for binary classification check

            # forward pass
            image = self.fabric.to_device(image)
            mask = self.fabric.to_device(mask)
            probs = self.__forward(image)
            
            # backward pass
            self.__backward(probs, mask.long(), train=False)

        self.model.train()

    def __forward(self, image : torch.Tensor) -> Tuple[torch.Tensor]:

        # combine image and coordinates
        probs = self.model(image)

        return probs
    
    def __backward(self, probs : torch.Tensor, mask : torch.Tensor, train : bool) -> torch.Tensor:

        # compute loss
        loss,classDice = self.loss_fn(mask, probs)

        if train:
            # optim step
            self.optim_step(loss)
            self.train_metrics.compute(mask, probs, loss.item(),classDice)
        else:
            self.validation_metrics.compute(mask, probs, loss.item(),classDice)