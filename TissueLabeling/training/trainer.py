import os

import lightning as L
import torch
import wandb
import math
from torch.utils.tensorboard import SummaryWriter

from TissueLabeling.metrics.metrics import Classification_Metrics
from TissueLabeling.training.logging import Log_Images
from TissueLabeling.utils import finish_wandb

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        fabric: L.Fabric,
        config,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.fabric = fabric
        # self.nr_of_classes = config.nr_of_classes
        # self.batch_size = config.batch_size
        # self.wandb_on = config.wandb_on
        # self.pretrained = config.pretrained
        # self.logdir = config.logdir
        # self.checkpoint_freq = config.checkpoint_freq
        # self.start_epoch = config.start_epoch + 1
        self.config = config

        if self.config.logdir:
            self.writer = SummaryWriter(self.config.logdir)
            print("SummaryWriter created")
        else:
            self.writer = None
        
        if self.fabric.global_rank == 0:
            self.image_logger = Log_Images(
                self.fabric,
                config=config,
                writer=self.writer,
            )

    def train_and_validate(self) -> None:
        self.train_metrics = Classification_Metrics(
            self.config.nr_of_classes, prefix="Train", wandb_on=self.config.wandb_on
        )
        self.validation_metrics = Classification_Metrics(
            self.config.nr_of_classes, prefix=f"Validation", wandb_on=self.config.wandb_on
        )

        print(
            f"Process {self.fabric.global_rank} starts training on {len(self.train_loader) // self.config.batch_size} batches per epoch over {self.config.num_epochs} epochs"
        )

        for epoch in range(self.config.start_epoch + 1, self.config.num_epochs + 1):
            self._train()

            self._validation()

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

    def _train(self) -> None:
        print('Training...')
        batch_idx = 0
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
        
    @torch.no_grad()
    def _validation(self) -> None:
        print('Validation...')
        self.model.eval()
        for i, (image, mask) in enumerate(self.val_loader):
            # mask[mask != 0] = 1 # uncomment for binary classification check

            # forward pass
            probs = self.model(image)

            # backward pass
            loss, classDice = self.loss_fn(mask.long(), probs)
            self.validation_metrics.compute(mask.long(), probs, loss.item(), classDice)
    
    def _log_metrics(self, epoch) -> None:
        if self.fabric.global_rank == 0:
            print(f'Process {self.fabric.global_rank} logging metrics...')
            self.train_metrics.log(epoch, commit=False, writer=self.writer)
            self.validation_metrics.log(epoch, commit=False, writer=self.writer)
    
    def _log_image(self, epoch) -> None:
        if self.fabric.global_rank == 0 and (epoch == 1 or epoch % self.config.image_log_freq == 0):
            print(f"Process {self.fabric.global_rank} saving image...")
            self.image_logger.logging(self.model, epoch, commit=True)
    
    def _reset_metrics(self) -> None:
        print("Resetting metrics...")
        self.train_metrics.reset()
        self.validation_metrics.reset()
    
    def _save_checkpoint(self, epoch) -> None:
        if self.config.save_checkpoint and (epoch == 1 or epoch % self.config.checkpoint_freq == 0):
            print(f"Saving epoch {epoch} checkpoint...")
            model_save_path = f"{self.config.logdir}/checkpoint_{epoch:04d}.ckpt"
            state = {
                "model": self.model,
                "optimizer": self.optimizer,
            }
            self.fabric.save(model_save_path, state)
    
    def _log_wandb(self,log) -> None:
        if self.fabric.global_rank == 0 and self.config.wandb_on:
            print("Saving state to wandb...")
            model_save_path = f"{self.config.logdir}/checkpoint_{self.config.num_epochs:04d}.ckpt"
            self._save_state(model_save_path,log)

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
            log (int): indicator if the model should be logged to wandb
        """

        if log:
            artifact = wandb.Artifact("Model", type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact)
