# -- Load some classes to produce spectral functions
from recan.factory import Parameter
from recan.factory import GaussianAnsatz
from recan.factory import NRQCDKernel

# -- Load the BaseModel class
from recan.models import BaseModel

# -- Load the factory isolated module
from factory import SPFactory

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Model(BaseModel):

    def __init__(self, input_size: int, output_size: int, name: str = ''):
        super().__init__(name)
        self.architecture = nn.Sequential(
            nn.Linear(input_size, 500), nn.ReLU(), nn.BatchNorm1d(500), nn.Dropout(0.1),
            nn.Linear(500, 500), nn.ReLU(), nn.BatchNorm1d(500), nn.Dropout(0.1),
            nn.Linear(500, 500), nn.ReLU(), nn.BatchNorm1d(500), nn.Dropout(0.1),
            nn.Linear(500, 500), nn.ReLU(), nn.BatchNorm1d(500), nn.Dropout(0.1),
            nn.Linear(500, 500), nn.ReLU(), nn.BatchNorm1d(500), nn.Dropout(0.1),
            nn.Linear(500, 500), nn.ReLU(), nn.BatchNorm1d(500), nn.Dropout(0.1),
            nn.Linear(500, 500), nn.ReLU(), nn.BatchNorm1d(500), nn.Dropout(0.1),
            nn.Linear(500, 500), nn.ReLU(), nn.BatchNorm1d(500), nn.Dropout(0.1),
            nn.Linear(500, output_size)
        )

if __name__ == '__main__':

    # Set the seed to be constant
    torch.manual_seed(916650397)

    # Generate the parameters
    param_A = Parameter('A', 0.1000, 1.0000)
    param_W = Parameter('W', 0.0100, 0.1000)
    param_M = Parameter('M', 0.1000, 3.5000)

    # Generate the gaussian ansatzs
    gauss = GaussianAnsatz(param_A, param_M, param_W)

    # Generate the kernel
    kernel = NRQCDKernel(64, 1000, [0.0, 5.0])

    # Generate the factory
    dataset = SPFactory([gauss for _ in range(5)], kernel)

    # Generate the data
    dataset.generate_data(10000, 64)

    # Wrap the dataset around a loader function
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # Generate a model to train
    model = Model(kernel.Nt, dataset.Ns).cuda()

    # Optimiser and learning rate scheduler
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

    # Transform the dataset to 0-1
    SC, TC = dataset.C.max() - dataset.C.min(), dataset.C.min()
    SL, TL = dataset.L.max() - dataset.L.min(), dataset.L.min()

    # Normalise the data
    dataset.C, dataset.L = (dataset.C - TC) / SC, (dataset.L - TL) / SL

    for epoch in range(100):

        # Track the total loss
        epoch_loss = []

        for nb, (C_data, L_data) in enumerate(loader):

            # Set the gradients to zero
            optim.zero_grad()

            # Move the data to the GPU
            C_data, L_data = C_data.cuda(), L_data.cuda()

            # Compute the model prediction
            L_pred = model(C_data)

            # Compute the loss function
            loss = (L_pred - L_data).pow(2).mean()

            # Append the loss
            epoch_loss += [loss]

            if (nb + 1) % 10 == 0:
                print(f'{nb + 1}: loss={loss:.6f}')

            # Backward pass and optimiser step
            loss.backward(), optim.step()

        epoch_loss = torch.tensor(epoch_loss)
        sched.step(epoch_loss.median())
        print(f'Epoch {epoch + 1}: loss_med={epoch_loss.median():.6f}')
