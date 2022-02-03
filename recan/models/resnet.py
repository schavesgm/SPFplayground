import torch.nn as nn
import torch

# Import the base network module. Allow executing this file.
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel

class ResidualBlock(nn.Module):
    """ Basic residual block composed by two convolutional layers. """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample = None):

        # Initialise the parents module
        super().__init__()

        # Generate the architecture of the block -- The block does not change L
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # Save the downsample method and the stride
        self.downsample, self.stride = downsample, stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Residual forward pass. The last part uses the residual addition. """

        # Now, produce the forward pass of the block
        out = self.layers(x)

        # If the downsample is available, then update it
        if self.downsample is not None:
            x = self.downsample(x)

        # Apply the ReLU and return
        return torch.relu(out + x)

def get_reslayer(num_layers: int, in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """ Generate a sequence of residual blocks. The starting block can contain a downsample
    layer to match the dimension of the Identity operation. The downsample layer is defined
    to be a convolution of one parameter per channel."""

    # In the first layer, it may be needed to downsample
    downsample = None

    # Try downsampling the image to match the dimensions with a conv1D
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )

    # List that will contain all the layers
    layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]

    # Now, concatenate more layers with input_C=output_C and stride=1
    layers += [ResidualBlock(out_channels, out_channels, 1) for _ in range(1, num_layers)]

    return nn.Sequential(*layers)

class ResidualNet(BaseModel):
    """ Basic implementation of a 18 layer Residual Network. """

    def __init__(self, input_size: int, output_size: int, name: str = ''):

        # Initialise the parent module
        super().__init__(name)

        # Unflatten the input data and batch normalise it
        self.augment = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Unflatten(1, (1, input_size)),
        )

        # Now, do the first convolution
        self.first_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Generate the residual network architecture
        self.resnet_layers = nn.Sequential(
            get_reslayer(2, 64, 64, 1),
            get_reslayer(2, 64, 128, 2),
            get_reslayer(6, 128, 256, 2),
            get_reslayer(3, 256, 512, 2),
        )

        # Output of the network using adaptative pool 1d
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d((1,)),
            nn.Flatten(1, -1),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_size),
        )

if __name__ == '__main__':
    pass
