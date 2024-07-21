"""
File: trainer.py
Author: Sabeen Lohawala
Date: 2024-04-09
Description: This file contains the Trainer, which implements the training loop.
"""
import os
import copy

import lightning as L
import torch
import wandb
import math
from torch.utils.tensorboard import SummaryWriter

from TissueLabeling.metrics.metrics import Classification_Metrics
from TissueLabeling.training.logging import Log_Images
from TissueLabeling.utils import finish_wandb

class Trainer:
    """
    Implements the training loop for the experiment.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        metric: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        fabric: L.Fabric,
        config,
    ) -> None:
        """
        Initializes the Trainer object.

        Args:
            model (torch.nn.Module): the PyTorch model to train
            train_loader (torch.utils.data.Dataloader): torch Dataloader for the training split
            val_loader (torch.utils.data.Dataloader): torch Dataloader for the validation split
            loss_fn (torch.nn.Module): loss function to use to train the model
            metric (torch.nn.Module): metric computed during each training and validation epoch to measure performance
            optimizer (torch.optim.Optimizer): optimizer used during training
            fabric (L.Fabric): fabric initialized so all objects can be sent to correct devices
            config (TissueLabeling.config.Configuration): contains the parameters for this run
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metric = metric
        self.optimizer = optimizer
        self.fabric = fabric
        self.config = config

        # output tensorboard logs to specified logdir
        if self.config.logdir:
            self.writer = SummaryWriter(self.config.logdir)
            print("SummaryWriter created")
        else:
            self.writer = None

        # only one GPU should log
        if self.fabric.global_rank == 0 and self.config.log_images:
            self.image_logger = Log_Images(
                self.fabric,
                config=config,
                writer=self.writer,
            )

    def train_and_validate(self) -> None:
        """
        This function implements the training and validation loop logic.
        """
        self.train_metrics = Classification_Metrics(
            self.config.nr_of_classes,
            prefix="Train",
            wandb_on=self.config.wandb_on,
            loss_name=self.config.loss_fn,
            metric_name=self.config.metric,
            class_specific_scores=self.config.class_specific_scores,
        )
        self.validation_metrics = Classification_Metrics(
            self.config.nr_of_classes,
            prefix=f"Validation",
            wandb_on=self.config.wandb_on,
            loss_name=self.config.loss_fn,
            metric_name=self.config.metric,
            class_specific_scores=self.config.class_specific_scores,
        )

        print(
            f"Process {self.fabric.global_rank} starts training on {len(self.train_loader)} batches per epoch over {self.config.num_epochs} epochs"
        )

        for epoch in range(self.config.start_epoch + 1, self.config.num_epochs + 1):
            self._train()

            self._validation()

            # sync loss and metrics across GPUs before logging
            self.fabric.barrier()
            self.train_metrics.sync(self.fabric)

            self._log_metrics(epoch)
            self._log_image(epoch)
            self._reset_metrics()

            # save model checkpoint
            self._save_checkpoint(epoch)

        if self.writer is not None:
            self.writer.close()

        self._log_wandb(log=False)
    
    def test(self) -> None:
        """
        This function computes the test dice metric for a specific experiment.
        """
        self.validation_metrics = Classification_Metrics(
            self.config.nr_of_classes,
            prefix=f"Test",
            wandb_on=self.config.wandb_on,
            loss_name=self.config.loss_fn,
            metric_name=self.config.metric,
            class_specific_scores=self.config.class_specific_scores,
        )

        print(
            f"Process {self.fabric.global_rank} starts getting test data"
        )

        self._validation()

        logging_dict = {
            f"{self.validation_metrics.prefix}/Loss/{self.validation_metrics.loss_name.title()}": sum(self.validation_metrics.loss)
            / len(self.validation_metrics.loss),
            f"{self.validation_metrics.prefix}/Metric/{self.validation_metrics.metric_name.title()}": sum(self.validation_metrics.metric)
            / len(self.validation_metrics.metric),
        }
        print(logging_dict)

        if self.writer is not None:
            self.writer.close()

    def _train(self) -> None:
        """
        This function implements the training loop within a single epoch.
        """
        print("Training...")
        self.model.train()
        for i, (image, mask) in enumerate(self.train_loader):
            # mask[mask != 0] = 1 # uncomment for binary classification check

            print(f"Process {self.fabric.global_rank}, batch {i}")

            self.optimizer.zero_grad()
            probs = self.model(image.to(torch.float32))
            loss = self.loss_fn(mask.long(), probs)
            self.fabric.backward(loss)
            self.optimizer.step()

            if self.config.class_specific_scores:
                overall_dice, class_dice = self.metric(mask.long(), probs)
            else:
                class_dice = None
                overall_dice = self.metric(mask.long(), probs)
            self.train_metrics.compute(
                loss=loss.item(), metric=overall_dice.item(), class_dice=class_dice
            )

    @torch.no_grad()
    def _validation(self) -> None:
        """
        This function implements the validation step within a single epoch.
        """
        print("Validation...")
        self.model.eval()
        for i, (image, mask) in enumerate(self.val_loader):
            # mask[mask != 0] = 1 # uncomment for binary classification check

            # forward pass
            probs = self.model(image)

            # backward pass
            # loss, classDice = self.loss_fn(mask.long(), probs)
            loss = self.loss_fn(mask.long(), probs)
            if self.config.class_specific_scores:
                overall_dice, class_dice = self.metric(mask.long(), probs)
            else:
                class_dice = None
                overall_dice = self.metric(mask.long(), probs)
            self.validation_metrics.compute(
                loss=loss.item(), metric=overall_dice.item(), class_dice=class_dice
            )

    def _log_metrics(self, epoch) -> None:
        """
        This function is used to log the train and validation loss and metrics for 
        the specified epoch to tensorboard.

        Args:
            epoch (int): the epoch for which these metrics are being logged
        """
        if self.fabric.global_rank == 0:
            print(f"Process {self.fabric.global_rank} logging metrics...")
            self.train_metrics.log(epoch, commit=False, writer=self.writer)
            self.validation_metrics.log(epoch, commit=False, writer=self.writer)

    def _log_image(self, epoch) -> None:
        """
        This function is used to log test images and the model outputs for the specified epoch to tensorboard.

        Args:
            epoch (int): the epoch for which these images are being logged.
        """
        if (
            self.config.log_images
            and self.fabric.global_rank == 0
            and (epoch == 1 or epoch % self.config.image_log_freq == 0)
        ):
            print(f"Process {self.fabric.global_rank} saving image...")
            self.image_logger.logging(self.model, epoch, commit=True)

    def _reset_metrics(self) -> None:
        """
        This method is used to reset the metrics.
        """
        print("Resetting metrics...")
        self.train_metrics.reset()
        self.validation_metrics.reset()

    def _save_checkpoint(self, epoch) -> None:
        """
        This method is used to determine whether to save a checkpoint of the model
        given the epoch.

        Args:
            epoch (int): the current epoch of training, used to determine whether to save the checkpoint.
        """
        if self.config.save_checkpoint and (
            epoch == 1 or epoch % self.config.checkpoint_freq == 0
        ):
            print(f"Saving epoch {epoch} checkpoint...")
            model_save_path = f"{self.config.logdir}/checkpoint_{epoch:04d}.ckpt"
            state = {
                "model": self.model,
                "optimizer": self.optimizer,
            }
            self.fabric.save(model_save_path, state)

    def _log_wandb(self, log) -> None:
        """
        This method is used to log data to Weights and Biases.

        Args:
            log (bool): flag to indicate whether to log current state to wandb
        """
        if self.fabric.global_rank == 0 and self.config.wandb_on:
            print("Saving state to wandb...")
            model_save_path = (
                f"{self.config.logdir}/checkpoint_{self.config.num_epochs:04d}.ckpt"
            )
            self._save_state(model_save_path, log)

            # add .out file to wandb and terminate
            wandb_log_file = os.path.join(
                os.getcwd(),
                "logs/job_train.out",
            )

            finish_wandb(wandb_log_file)

    def _save_state(
        self,
        path,
        log: bool = False,
    ) -> None:
        """
        Save the pytorch model and the optimizer state

        Args:
            path (str): where to save the state
            log (int): flag to indicate if the model should be logged to wandb
        """
        if log:
            artifact = wandb.Artifact("Model", type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)
