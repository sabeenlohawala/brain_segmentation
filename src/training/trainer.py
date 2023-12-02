
import torch
import webdataset as wds
import lightning as L
import wandb
from collections import Counter
from typing import Tuple
import numpy as np

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
            batch_size : int,
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
        self.batch_size = batch_size

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

        # state = {
        #     'epoch': epoch,
        #     'batch_idx': batch_idx,
        #     'model': self.model,
        #     'optimizer': self.optimizer,
        #     }
        
        # self.fabric.save(PATH, state)
        # torch.save(
        #             self.model.state_dict(),
        #             PATH,
        #         )

        if log:
            artifact = wandb.Artifact('Model', type='model') # comment to not save to wandb
            artifact.add_file(PATH) # comment to not save to wandb
            wandb.log_artifact(artifact) # comment to not save to wandb
            # pass

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

        print(f"Process {self.fabric.global_rank} starts training on {len(self.train_loader) // self.batch_size} batches per epoch over {epochs} epochs")

        for epoch in range(epochs):
            self.model.train()
            for i, (image, mask) in enumerate(self.train_loader):
                # mask[mask != 0] = 1 # uncomment for binary classification check

                print(self.fabric.global_rank, i)
                image = image.repeat((1,3,1,1)) # uncomment if pretrained = True

                self.optimizer.zero_grad()
                probs = self.model(image)
                loss, classDice = self.loss_fn(mask.long(), probs)
                self.fabric.backward(loss)
                self.optimizer.step()
                self.train_metrics.compute(mask.long(), probs, loss.item(), classDice)

                batch_idx += 1
                # print(batch_idx)
                
            print(f"Process {self.fabric.global_rank} finished epoch {epoch}...")

            # evaluation

            print("start validation...")
            # compute loss on validation data
            self._validation(self.val_loader)
            # log loss and reset
            # print(f"Process {self.fabric.global_rank} reached barrier")
            # self.fabric.barrier()
            # print(f"Process {self.fabric.global_rank} passed barrier")
            # self.train_metrics.sync(self.fabric)
            # self.validation_metrics.sync(self.fabric)
            # print(f"Process {self.fabric.global_rank} synced tensors")

            if self.fabric.global_rank == 0:

                print("Rank:", self.fabric.global_rank)

                self.train_metrics.log(commit=False)
                self.validation_metrics.log(commit=False)

                print("saving image...")        
                self.image_logger.logging(self.model, epoch, commit=True)

            self.train_metrics.reset()
            self.validation_metrics.reset()

        PATH = "/home/sabeen/brain_segmentation/models/checkpoint.ckpt"
        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model': self.model,
            'optimizer': self.optimizer,
            }
        self.fabric.save(PATH, state)
        # torch.save(
        #             self.model.state_dict(),
        #             PATH,
        #         )

        if self.fabric.global_rank == 0:

            print("final saving model state...")
            self._save_state(epoch, batch_idx, log=False)

            # add .out file to wandb and terminate
            self.finish_wandb('/om2/user/sabeen/brain_segmentation/jobs/job_train.out') # comment to not save to wandb

    @torch.no_grad()
    def _validation(self, data_loader : wds.WebLoader) -> None:

        self.model.eval()
        for i, (image, mask) in enumerate(data_loader):
            # mask[mask != 0] = 1 # uncomment for binary classification check

            # forward pass
            image = image.repeat((1,3,1,1)) # uncomment if pretrained = True
            probs = self.model(image)
            
            # backward pass
            # self.__backward(probs, mask.long(), train=False)
            loss, classDice = self.loss_fn(mask.long(), probs)
            self.validation_metrics.compute(mask.long(), probs, loss.item(), classDice)

        # self.model.train()

    # def __forward(self, image : torch.Tensor) -> Tuple[torch.Tensor]:

    #     # combine image and coordinates
    #     probs = self.model(image)

    #     return probs
    
    # def __backward(self, probs : torch.Tensor, mask : torch.Tensor, train : bool) -> torch.Tensor:

    #     # compute loss
    #     loss, dice_coeff, classDice = self.loss_fn(mask, probs)

    #     if train:
    #         # optim step
    #         self.optim_step(loss)
    #         self.train_metrics.compute(mask, probs, loss.item(), dice_coeff, classDice)
    #     else:
    #         self.validation_metrics.compute(mask, probs, loss.item(), dice_coeff, classDice)
# import torch
# import webdataset as wds
# import lightning as L
# import wandb
# from collections import Counter
# from typing import Tuple

# from training.logging import Log_Images
# from models.metrics import Classification_Metrics

# class Trainer():

#     def __init__(
#             self,
#             model : torch.nn.Module,
#             nr_of_classes: int,
#             train_loader, #: wds.WebLoader,
#             val_loader, #: wds.WebLoader,
#             loss_fn : torch.nn.Module,
#             optimizer : torch.optim.Optimizer,
#             fabric : L.Fabric,
#             save_every : int = 100000,
#             ) -> None:
        
#         self.model = model
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
#         self.fabric = fabric
#         self.save_every = save_every
#         self.nr_of_classes = nr_of_classes

#         if self.fabric.global_rank == 0:
#             self.image_logger = Log_Images(self.fabric, nr_of_classes=nr_of_classes)

#     def _save_state(self, epoch : int, batch_idx : int, log: bool = False) -> None:
#         '''
#         Save the pytorch model and the optimizer state

#         Args:
#             epoch (int): epoch number
#             batch_idx (int): training batch index
#             log (int): indicator if the model should be logged to wandb
#         '''

#         PATH = "/home/sabeen/brain_segmentation/models/checkpoint.ckpt"

#         state = {
#             'epoch': epoch,
#             'batch_idx': batch_idx,
#             'model': self.model,
#             'optimizer': self.optimizer,
#             }
        
#         self.fabric.save(PATH, state)

#         if log:
#             artifact = wandb.Artifact('Model', type='model')
#             artifact.add_file(PATH)
#             wandb.log_artifact(artifact)

#     def optim_step(self, loss : torch.Tensor) -> None:
#         '''
#         One step of the optimizer

#         Args:
#             loss (torch.Tensor): computed loss for minibatch
#         '''

#         # faster than optimizer.zero_grad()
#         for param in self.model.parameters():
#             param.grad = None

#         # optimizer step
#         if torch.isnan(loss) or torch.isinf(loss):
#             raise ValueError("Loss is NaN or Inf")
#         else:
#             self.fabric.backward(loss)
#             self.optimizer.step()

#     def finish_wandb(self, out_file : str) -> None:
#         '''
#         Finish Weights and Biases

#         Args:
#             out_file (str): name of the .out file of the run
#         '''

#         # add .out file to wandb
#         artifact_out = wandb.Artifact('OutFile', type='out_file')
#         artifact_out.add_file(out_file)
#         wandb.log_artifact(artifact_out)         
#         # finish wandb
#         wandb.finish(quiet=True)

#     def train(self, epochs : int) -> None:
        
#         batch_idx = 0

#         self.train_metrics = Classification_Metrics(self.nr_of_classes, prefix="Train")
#         self.validation_metrics = Classification_Metrics(self.nr_of_classes, prefix=f"Validation")

#         # uncomment after MNIST
#         print(f"Process {self.fabric.global_rank} starts training on {self.train_loader.length} batches per epoch over {epochs} epochs")

#         for epoch in range(epochs):

#             for image, mask, _ in self.train_loader:
#                 # import pdb
#                 # pdb.set_trace()

#                 # print(torch.unique(mask))
#                 # mask[mask != 0] = 1 # uncomment for binary classification check

#                 image = self.fabric.to_device(image)
#                 mask = self.fabric.to_device(mask)
#                 probs = self.__forward(image)
#                 self.__backward(probs, mask.long(), train=True)

#                 batch_idx += 1
#                 print(batch_idx)

#             # MNIST
#             # n_total_steps = 12000 # len(self.train_loader)
#             # n_correct = 0
#             # n_total_steps = 12000
#             # self.model.train()
#             # # n_samples = len(self.val_loader.dataset)
#             # for i, (images, labels) in enumerate(self.train_loader):
#             #     # origin shape: [100, 1, 28, 28]
#             #     # resized: [100, 784]
#             #     # reshape to match dimension of first linear layer
#             #     images = images.reshape(-1, 28*28)#.to(device)
#             #     labels = labels#.to(device)
#             #     # images = fabric.to_device(images.reshape(-1, 28*28))#.to(device)
#             #     # labels = fabric.to_device(labels)#.to(device)

#             #     # Forward pass and loss calculation
#             #     # [100, 10]
#             #     outputs = self.model(images)
#             #     loss = self.loss_fn(outputs, labels)

#             #     # _, predicted = torch.max(outputs, 1)
#             #     # # print(predicted == labels)
#             #     # n_correct += (predicted == labels).sum().item()

#             #     # Backward and optimize
#             #     # loss.backward()
#             #     self.fabric.backward(loss)
#             #     self.optimizer.step()
#             #     self.optimizer.zero_grad()

#             #     if (i+1) % 100 == 0:
#             #         print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
#             # train_acc = n_correct / n_samples

#             print(f"Process {self.fabric.global_rank} finished epoch {epoch}...")

#             # evaluation

#             print("start validation...")
#             # compute loss on validation data
#             self._validation(self.val_loader)
#             # log loss and reset
#             print(f"Process {self.fabric.global_rank} reached barrier")
#             self.fabric.barrier()
#             print(f"Process {self.fabric.global_rank} passed barrier")
#             self.train_metrics.sync(self.fabric)
#             self.validation_metrics.sync(self.fabric)
#             print(f"Process {self.fabric.global_rank} synced tensors")

#             # MNIST
#             # self.model.eval()
#             # with torch.no_grad():
#             #     n_correct = 0
#             #     n_samples = len(self.val_loader.dataset)
#             #     count = 0

#             #     for images, labels in self.val_loader:
#             #         # [100, 784]
#             #         images = images.reshape(-1, 28*28)#.to(device)
#             #         # [100]
#             #         labels = labels#.to(device)

#             #         # # [100, 784]
#             #         # images = fabric.to_device(images.reshape(-1, 28*28))#.to(device)
#             #         # # [100]
#             #         # labels = fabric.to_device(labels)#.to(device)

#             #         # [100, 10]
#             #         outputs = self.model(images)

#             #         # max returns (output_value ,index)
#             #         # [100]
#             #         _, predicted = torch.max(outputs, 1)
#             #         # print(predicted == labels)
#             #         n_correct += (predicted == labels).sum().item()
#             #         count += images.shape[0]
#             #     # print('count', count)

#             #     acc = n_correct / count
#             #     print(f'Accuracy of the network on the {count} test images: {100*acc} %')
#             # self.fabric.barrier()
#             # count = self.fabric.all_reduce(count,reduce_op='sum')
#             # n_correct = self.fabric.all_reduce(n_correct,reduce_op='sum')
#             # total_acc = n_correct / count

#             if self.fabric.global_rank == 0:

#                 print("Rank:", self.fabric.global_rank)

#                 self.train_metrics.log(commit=False)
#                 self.validation_metrics.log(commit=False)

#                 print("saving image...")        
#                 self.image_logger.logging(self.model, epoch, commit=True)

#             # MNIST
#             # if self.fabric.global_rank == 0:
#             #     logging_dict = {
#             #         # f"Train/Accuracy": train_acc,
#             #         f"Val/Accuracy": total_acc,
#             #         # f"Assert": self.Assert,
#             #     }

#             #     wandb.log(logging_dict, commit=True)

#             self.train_metrics.reset()
#             self.validation_metrics.reset()

#         if self.fabric.global_rank == 0:

#             print("final saving model state...")
#             self._save_state(epoch, batch_idx, log=True)

#             # add .out file to wandb and terminate
#             self.finish_wandb('/om2/user/sabeen/brain_segmentation/jobs/job_train.out')

#     @torch.no_grad()
#     def _validation(self, data_loader : wds.WebLoader) -> None:

#         self.model.eval()
#         for image, mask, _ in data_loader:
#             # mask[mask != 0] = 1 # uncomment for binary classification check

#             # forward pass
#             image = self.fabric.to_device(image)
#             mask = self.fabric.to_device(mask)
#             probs = self.__forward(image)
            
#             # backward pass
#             self.__backward(probs, mask.long(), train=False)

#         self.model.train()

#     def __forward(self, image : torch.Tensor) -> Tuple[torch.Tensor]:

#         # combine image and coordinates
#         probs = self.model(image)

#         return probs
    
#     def __backward(self, probs : torch.Tensor, mask : torch.Tensor, train : bool) -> torch.Tensor:

#         # compute loss
#         loss,classDice = self.loss_fn(mask, probs)
#         # print('compute loss:', self.fabric.global_rank, dice, loss)
#         # import pdb
#         # pdb.set_trace()

#         if train:
#             # optim step
#             self.optim_step(loss)
#             self.train_metrics.compute(mask, probs, loss.item(), classDice)
#         else:
#             self.validation_metrics.compute(mask, probs, loss.item(), classDice)



        