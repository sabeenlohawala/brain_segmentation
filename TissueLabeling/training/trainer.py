import os

import lightning as L
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from TissueLabeling.models.metrics import Classification_Metrics
from TissueLabeling.training.logging import Log_Images


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        # nr_of_classes: int,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        fabric: L.Fabric,
        # batch_size: int,
        # wandb_on: bool,
        # pretrained: bool,
        config,
        # logdir: str = "",
        save_every: int = 100000,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.fabric = fabric
        self.save_every = save_every
        self.nr_of_classes = config.nr_of_classes
        self.batch_size = config.batch_size
        self.wandb_on = config.wandb_on
        self.pretrained = config.pretrained
        self.logdir = config.logdir

        if self.fabric.global_rank == 0:
            self.image_logger = Log_Images(
                self.fabric,
                config=config,
            )

        if self.logdir:
            if not os.path.isdir(self.logdir):
                os.makedirs(self.logdir)
            self.writer = SummaryWriter(self.logdir)
            print("SummaryWriter created")
        else:
            self.writer = None

    def train(self, epochs: int) -> None:
        batch_idx = 0

        self.train_metrics = Classification_Metrics(
            self.nr_of_classes, prefix="Train", wandb_on=self.wandb_on
        )
        self.validation_metrics = Classification_Metrics(
            self.nr_of_classes, prefix=f"Validation", wandb_on=self.wandb_on
        )

        print(
            f"Process {self.fabric.global_rank} starts training on {len(self.train_loader) // self.batch_size} batches per epoch over {epochs} epochs"
        )

        for epoch in range(epochs):
            self.model.train()
            for i, (image, mask) in enumerate(self.train_loader):
                # mask[mask != 0] = 1 # uncomment for binary classification check

                print(f"Process {self.fabric.global_rank}, batch {i}")

                self.optimizer.zero_grad()
                probs = self.model(image)
                loss, classDice = self.loss_fn(mask.long(), probs)
                self.fabric.backward(loss)
                self.optimizer.step()
                self.train_metrics.compute(mask.long(), probs, loss.item(), classDice)

                batch_idx += 1

            print(f"Process {self.fabric.global_rank} finished epoch {epoch}...")

            # evaluation
            print("start validation...")
            # compute loss on validation data
            self._validation(self.val_loader)
            self.fabric.barrier()
            self.train_metrics.sync(self.fabric)

            if self.fabric.global_rank == 0:
                print("Rank:", self.fabric.global_rank)

                self.train_metrics.log(commit=False, writer=self.writer)
                self.validation_metrics.log(commit=False, writer=self.writer)

                print("saving image...")
                self.image_logger.logging(self.model, epoch, commit=True)

            self.train_metrics.reset()
            self.validation_metrics.reset()

        # save model and log to wandb
        model_save_path = f"{self.logdir}/checkpoint.ckpt"
        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model": self.model,
            "optimizer": self.optimizer,
        }
        self.fabric.save(model_save_path, state)

        if self.writer is not None:
            self.writer.close()

        if self.fabric.global_rank == 0 and self.wandb_on:
            print("final saving model state...")
            self._save_state(epoch, batch_idx, log=False, path=model_save_path)

            # add .out file to wandb and terminate
            self.finish_wandb("/om2/user/sabeen/brain_segmentation/jobs/job_train.out")

    @torch.no_grad()
    def _validation(self, data_loader) -> None:
        self.model.eval()
        for i, (image, mask) in enumerate(data_loader):
            # mask[mask != 0] = 1 # uncomment for binary classification check

            # forward pass
            probs = self.model(image)

            # backward pass
            loss, classDice = self.loss_fn(mask.long(), probs)
            self.validation_metrics.compute(mask.long(), probs, loss.item(), classDice)

    def _save_state(
        self,
        epoch: int,
        batch_idx: int,
        log: bool = False,
        path="/home/sabeen/brain_segmentation/models/checkpoint.ckpt",
    ) -> None:
        """
        Save the pytorch model and the optimizer state

        Args:
            epoch (int): epoch number
            batch_idx (int): training batch index
            log (int): indicator if the model should be logged to wandb
        """

        if log:
            artifact = wandb.Artifact("Model", type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def finish_wandb(self, out_file: str) -> None:
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
