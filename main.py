import os
import time

# -- Load some classes to produce spectral functions
from recan.factory import Parameter
from recan.factory import GaussianAnsatz
from recan.factory import NRQCDKernel

# -- Load the BaseModel class
from recan.models import BaseModel
# from recan.models import ResidualNet

# -- Load the factory isolated module
from factory import SPFactory

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
plt.style.use(['science', 'ieee', 'monospace'])

# TODO: Seems that loss_R and loss_C do not change anything.
# TODO: Add some dropout to check if it enhances training -- Regularisation

class Model(BaseModel):

    # CHANGES:
    # -- Added Dropout with p=0.1 after all ReLU -> Check performance?

    def __init__(self, input_size: int, output_size: int, name: str = ''):

        super().__init__(name)

        # Generate the architecture: This one works at least for 2peaks only
        self.arch = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, output_size)
        )

if __name__ == '__main__':

    # Set the seed to be constant
    torch.manual_seed(916650397)

    # Number of peaks to be used in any spectral function
    num_peaks = 2

    # Generate the parameters
    param_A = Parameter('A', 0.1000, 1.0000)
    param_W = Parameter('W', 0.0100, 0.1000)
    param_M = Parameter('M', 0.1000, 3.5000)

    # Generate the gaussian ansatzs
    gauss = GaussianAnsatz(param_A, param_M, param_W)

    # Generate the kernel
    kernel = NRQCDKernel(64, 1000, [0.0, 5.0])

    # Generate the factory
    dataset = SPFactory([gauss for _ in range(num_peaks)], kernel)

    # Generate the data: Increased from 64 to 128, check performance
    data = dataset.generate_data(100000, 128)

    # Get the normalisation constants for the input and label
    SC, TC = data.C.std(), data.C.mean()
    SL, TL = data.L.std(), data.L.mean()

    # Wrap the dataset around a loader function
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # Generate a model to train
    model = Model(dataset.Nt, dataset.Ns).cuda()

    # Optimiser and learning rate scheduler
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=15, factor=0.5, min_lr=1e-5)

    # Get a random example
    ex = torch.randint(0, dataset.Nb, (1, ))

    # Scalers for each loss function
    scale_L, scale_C, scale_R = 1.0, 100, 1.0

    # Train the model for different number of epochs
    for epoch in range(500):

        # Make the model trainable
        model.train()

        # Track the total loss and save the initial time
        epoch_loss, start = [], time.time()

        # Iterate through all minibatches
        for nb, data in enumerate(loader):

            # Set the gradients to zero
            optim.zero_grad()

            # Move the data to the GPU.
            C_data, L_data = data.C.flatten(1, -1).cuda(), data.L.flatten(1, -1).cuda()

            # Normalise the input and label data
            C_data, L_data = (C_data - TC) / SC, (L_data - TL) / SL

            # Compute the prediction of the network -> The output will be normalised
            L_pred = model(C_data)

            # Compute the loss functions
            loss = (L_pred - L_data).pow(2).mean()

            # Append the loss to the control tensor
            epoch_loss += [loss]

            # Backward pass and optimiser step
            loss.backward(), optim.step()

        epoch_loss = torch.tensor(epoch_loss)
        sched.step(epoch_loss.median())

        # Get the learning rate
        lr = optim.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}: loss_med={epoch_loss.median():.6f}, lr={lr:.6f}, eta={time.time() - start}')

        if (epoch + 1) % 10 == 0:

            # Generate the prediction of the model
            with torch.no_grad():

                # Set the network in evaluation mode
                model.eval()

                # Figure to plot the data
                fig = plt.figure(figsize=(6, 4))

                # Add some axis to the figure
                axis = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

                # References to the axis
                aL, aR, aC = axis

                # Set some parameters in each of the axis
                aL.set(xlabel=r'$n_s$', ylabel=r'$L(n_s)$')
                aR.set(xlabel=r'$\omega$', ylabel=r'$\rho(\omega)$')
                aC.set(xlabel=r'$\tau$', ylabel=r'$C(\tau)$', yscale='log')

                # Get the label coefficients from the dataset
                L_data = dataset.L[ex, :].view(1, dataset.Ns)

                # Generate the label objects
                data = dataset.reconstruct(L_data.cpu())

                # Compute the predicted coefficients
                L_pred = SL * model((data.C.cuda() - TC) / SC).cpu() + TL

                # Reconstruct the predicted objects -> Un-normalised output
                pred = dataset.reconstruct(L_pred)

                # Plot the coefficients
                aL.plot(data.L.flatten(), color='blue')
                aL.plot(pred.L.flatten(), color='red')

                # Plot the spectral functions
                aR.plot(kernel.omega, data.R.flatten(), color='blue')
                aR.plot(kernel.omega, pred.R.flatten(), color='red')

                # Plot the correlation functions
                aC.plot(kernel.tau, data.C.flatten(), color='blue')
                aC.plot(kernel.tau, pred.C.flatten(), color='red')

                # Save the data
                if not os.path.exists('./figures'): os.makedirs('./figures')
                fig.tight_layout()
                fig.savefig(f'./figures/epoch_{epoch + 1}.pdf')

                # Close the figures
                plt.cla(), plt.clf(), plt.close(fig)
