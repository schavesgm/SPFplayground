import os
import time

# -- Load some classes to produce spectral functions
from recan.factory import Parameter
from recan.factory import GaussianAnsatz
from recan.factory import NRQCDKernel

# -- Load the BaseModel class
from recan.models import BaseModel
from recan.models import ResidualNet

# -- Load the factory isolated module
from factory import SPFactory

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
plt.style.use(['science', 'ieee', 'monospace'])

# TODO: Add several examples to plot instead of just one.

class StandardScaler:
    """ StandardScaler class that scales and descales tensors using standard normal. """

    def __init__(self, data: torch.Tensor):
        # Compute the mean and the standard deviation of the data provided
        self.mu, self.sigma = data.mean(), data.std()

    def scale(self, data: torch.Tensor) -> torch.Tensor:
        """ Transform data: X -> ZX = (X - mu) / sigma. """
        return (data - self.mu) / self.sigma

    def descale(self, data: torch.Tensor) -> torch.Tensor:
        """ Transform data: ZX -> X = (ZX * sigma) + mu. """
        return data * self.sigma + self.mu

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
    kernel = NRQCDKernel(64, 1000, [0.0, 8.0])

    # Generate the factory
    dataset = SPFactory([gauss for _ in range(num_peaks)], kernel)

    # Generate the data: Increased from 64 to 128, check performance
    data = dataset.generate_data(50000, 64)

    # Generate two scalers for the data
    C_scaler, L_scaler = StandardScaler(data.C.log()), StandardScaler(data.L)

    # Wrap the dataset around a loader function
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # Generate a model to train
    model = ResidualNet(dataset.Nt, dataset.Ns, 'ResNet').cuda()

    # Optimiser and learning rate scheduler
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=20, factor=0.5, min_lr=1e-5, cooldown=10
    )

    # Get some random examples
    ex = torch.randint(0, dataset.Nb, (4,))

    # Scalers for each loss function
    scale_L, scale_C, scale_R = 1.0, 100, 1.0

    # Train the model for different number of epochs
    for epoch in range(500):

        # Make the model trainable
        model.train()

        # Track the total loss and save the initial time
        epoch_loss, start = [], time.time()

        # Iterate through all minibatches
        for nb, (C_data, L_data) in enumerate(loader):

            # Set the gradients to zero
            optim.zero_grad()

            # Normalise the input and label data and move to the GPU
            C_data, L_data = C_scaler.scale(C_data.log()).cuda(), L_scaler.scale(L_data).cuda()

            # Compute the prediction of the network -> The output will be normalised
            L_pred = model(C_data)

            # Compute the loss functions
            loss = (L_pred - L_data).pow(2).mean()

            # Append the loss to the control tensor
            epoch_loss += [loss]

            # Backward pass and optimiser step
            loss.backward(), optim.step()

        epoch_loss = torch.tensor(epoch_loss)
        sched.step(epoch_loss.mean())

        # Get the learning rate
        lr = optim.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}: loss={epoch_loss.mean():.6f}, lr={lr:.6f}, eta={time.time() - start}')

        if (epoch + 1) % 10 == 0:

            # Generate the prediction of the model
            with torch.no_grad():

                # Set the network in evaluation mode
                model.eval()

                # Generate a figure to plot the data
                fig = plt.figure(figsize=(10, 8))

                # Add several axis to the figure
                axis = [fig.add_subplot(ex.shape[0], 2, i) for i in range(1, 2 * ex.shape[0] + 1)]

                # Get the label coefficients and the correlation functions
                L_data, C_data = dataset.L[ex, :], dataset.C[ex, :]

                # Compute the predicted coefficients from the scaled correlation functions
                L_pred = L_scaler.descale(model(C_scaler.scale(C_data.cuda().log()))).cpu()

                # Generate the label and the predicted objects
                data, pred = dataset.reconstruct(L_data), dataset.reconstruct(L_pred)

                # Iterate through all the examples to plot them
                for e in range(ex.shape[0]):

                    # Select the corresponding axis
                    aL, aR, = axis[2 * e: 2 * (e + 1)]

                    # Set some parameters in each of the axis
                    aL.set(xlabel=r'$n_s$', ylabel=r'$L(n_s)$')
                    aR.set(xlabel=r'$\omega$', ylabel=r'$\rho(\omega)$')

                    # Plot the coefficients
                    aL.plot(L_data[e, :], color='blue')
                    aL.plot(L_pred[e, :], color='red')

                    # Plot the spectral functions
                    aR.plot(kernel.omega, data.R[e, :], color='blue')
                    aR.plot(kernel.omega, pred.R[e, :], color='red')

                # Make the plot nicer
                fig.tight_layout()

                # Save the data in the corresponding folder
                folder = f'./figures/{model.name}/p{num_peaks}_s{dataset.Ns}_b{dataset.Nb}'
                if not os.path.exists(folder): os.makedirs(folder)
                fig.savefig(os.path.join(folder, f'epoch_{epoch + 1}.pdf'))

                # Close the figures
                plt.cla(), plt.clf(), plt.close(fig)
