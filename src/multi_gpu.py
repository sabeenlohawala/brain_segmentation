
import torch
from typing import Tuple
import webdataset as wds
import lightning as L

from utils import set_seed, init_cuda, init_fabric
from models.metrics import Dice
from models.segformer import Segformer
from data.dataset import get_data_loader, mapping

NR_OF_CLASSES = 116
BATCH_SIZE = 64
LEARNING_RATE = 3e-6
N_EPOCHS = 1
DATASET = 'medium'
MODEL_NAME = "segformer"
SEED = 42
SAVE_EVERY = "epoch"
MASK_MAPPING = True
PRECISION = '32-true' #"16-mixed"

def main():
    set_seed(SEED)

    fabric = init_fabric(precision=PRECISION) # accelerator="gpu", devices=2, num_nodes=1
    init_cuda()

    # model
    model = Segformer(NR_OF_CLASSES)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # loss function
    loss_fn = Dice(NR_OF_CLASSES)

    model, optimizer = fabric.setup(model, optimizer)
    # if A100 GPU, compile model for HUGE speed up
    if "A100" in torch.cuda.get_device_name():
        print("Compile model...")
        model = torch.compile(model)

    train_loader, val_loader, _ = get_data_loader(DATASET, batch_size=BATCH_SIZE)

    trainer = Trainer(
        model=model,
        nr_of_classes=NR_OF_CLASSES,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        fabric=fabric,
        mask_mapping=MASK_MAPPING,
    )
    trainer.train(N_EPOCHS)

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
            mask_mapping: bool = False,
            ) -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.fabric = fabric
        self.save_every = save_every
        self.nr_of_classes = nr_of_classes

        # if self.fabric.global_rank == 0:
        #     self.image_logger = Log_Images(self.fabric, nr_of_classes=nr_of_classes, mask_mapping=mask_mapping)

    def _save_state(self, epoch : int, batch_idx : int) -> None:
        '''
        Save the pytorch model and the optimizer state

        Args:
            epoch (int): epoch number
            batch_idx (int): training batch index
        '''

        PATH = "/home/matth406/brain_segmentation/models/checkpoint.ckpt"

        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model': self.model,
            'optimizer': self.optimizer,
            }
        
        self.fabric.save(PATH, state)

        # artifact = wandb.Artifact('Model', type='model')
        # artifact.add_file(PATH)
        # wandb.log_artifact(artifact)

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

    def train(self, epochs : int) -> None:
        
        batch_idx = 0

        # self.train_metrics = Classification_Metrics(self.nr_of_classes)
        # self.validation_metrics = Classification_Metrics(self.nr_of_classes)

        print(f"Process {self.fabric.global_rank} starts training on {self.train_loader.length} batches per epoch over {epochs} epochs")

        for epoch in range(epochs):

            for i, (image, mask, _) in enumerate(self.train_loader):

                # forward pass
                mask = mapping(mask)
                image = self.fabric.to_device(image)
                mask = self.fabric.to_device(mask)
                probs = self.__forward(image)
                
                # backward pass
                self.__backward(probs, mask.long(), train=True)

                batch_idx += 1

                print(i)
                if i == 10:
                    break

            print(f"Process {self.fabric.global_rank} finished epoch {epoch}...")

            # evaluation

            # save model and optimizer state
            print("save checkpoint...")
            if self.fabric.global_rank == 0:
                self._save_state(epoch, batch_idx)

            print("start validation...")
            # compute loss on validation data
            self._validation(self.val_loader)
            # log loss and reset
            # self.train_metrics.log_and_reset("Train", self.fabric, commit=False)
            # self.validation_metrics.log_and_reset("Validation", self.fabric, commit=True)

        print("saving image...")
        self.image_logger.logging(self.model, epoch)

        # print("final saving image and model state...")
        # self.image_logger.logging(self.model, epoch)
        # self._save_state(epoch, batch_idx)

        # add .out file to wandb and terminate
        # self.finish_wandb('/home/matth406/brain_segmentation/jobs/job.out')

    @torch.no_grad()
    def _validation(self, data_loader : wds.WebLoader) -> None:

        self.model.eval()

        for i, (image, mask, _) in enumerate(data_loader):

            # forward pass
            mask = mapping(mask)
            image = self.fabric.to_device(image)
            mask = self.fabric.to_device(mask)
            probs = self.__forward(image)
            
            # backward pass
            self.__backward(probs, mask.long(), train=False)

            print(i)
            if i == 10:
                break

        self.model.train()

    def __forward(self, image : torch.Tensor) -> Tuple[torch.Tensor]:

        # combine image and coordinates
        probs = self.model(image)

        return probs
    
    def __backward(self, probs : torch.Tensor, mask : torch.Tensor, train : bool) -> torch.Tensor:

        # compute loss
        loss = self.loss_fn(mask, probs)

        if train:
            # optim step
            self.optim_step(loss)
            # self.train_metrics.compute(mask, probs, loss.item())
        # else:
            # self.validation_metrics.compute(mask, probs, loss.item())

    
    # # train
    # for i, (image, mask,_) in enumerate(train_loader):

    #     mask = mapping(mask)
    #     image = fabric.to_device(image)
    #     mask = fabric.to_device(mask)

    #     probs = model(image)

    #     loss = loss_fn(mask.long(), probs)

    #     # faster than optimizer.zero_grad()
    #     for param in model.parameters():
    #         param.grad = None

    #     # optimizer step
    #     if torch.isnan(loss) or torch.isinf(loss):
    #         raise ValueError("Loss is NaN or Inf")
    #     else:
    #         fabric.backward(loss)
    #         optimizer.step()

    #     print(i)
    #     if i == 100:
    #         break

    # print(f"Process {fabric.global_rank} finished training")

    # # validation
    # for i, (image,mask,_) in enumerate(val_loader):

    #     mask = mapping(mask)
    #     image = fabric.to_device(image)
    #     mask = fabric.to_device(mask)

    #     probs = model(image)

    #     loss = loss_fn(mask.long(), probs)

    #     print(i)
    #     if i == 100:
    #         break

    # print(f"Process {fabric.global_rank} finished validation")

if __name__ == "__main__":
    main()