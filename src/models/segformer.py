
import torch
from torch import nn
from torchinfo import summary

from transformers import SegformerForSemanticSegmentation, SegformerConfig

# segformer model
class Segformer(nn.Module):

    def __init__(self, nr_of_classes: int, pretrained: bool = False):
        '''
        Initialize Segformer mit-b1 calibration

        Args:
            nr_of_classes (int): number of input classes
            pretrained (bool, optional): use transfer learning. Defaults to False.
        '''
        super().__init__()

        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b1")
            self.segformer.decode_head.classifier = nn.Conv2d(256, nr_of_classes, kernel_size=(1, 1), stride=(1, 1))

        else:
            config = SegformerConfig(num_channels=1, num_labels=nr_of_classes, hidden_sizes=[64, 128, 320, 512])
            self.segformer = SegformerForSemanticSegmentation(config)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x : torch.tensor):
        
        self.logits = self.segformer(x).logits
        self.logits = torch.nn.functional.interpolate(self.logits, size=(162, 194), mode='bilinear')
        self.probs = self.softmax(self.logits)

        return self.probs

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Segformer(51, pretrained=True)

    summary(
        model,
        input_size=[(512+256+128+64, 3, 162, 194)],
        col_names=["input_size", "output_size", "num_params"],
    )