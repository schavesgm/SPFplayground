# -- Import some built-in modules
import os
import time
import argparse
from pathlib import Path

# -- Import some third-party modules
import torch
import matplotlib
import matplotlib.pyplot as plt

# -- Import some user-defined modules
from recan.factory import Parameter
from recan.factory import GaussianAnsatz
from recan.factory import NRQCDKernel
from recan.factory import SPFactory
from recan.models import ResidualNet

# -- Use this plotting backend to avoid memory leaks
matplotlib.use('Agg')

def parse_arguments() -> argparse.Namespace:
    """ Parse some command line arguments. """

    # Generate a parser to take command line arguments
    parser = argparse.ArgumentParser('SPFplayground')

    # Add some parameters to the parser
    parser.add_argument('-Nb', type=int, help='Number of training examples.')
    parser.add_argument('-Ns', type=int, help='Number of coefficients in the expansion')
    parser.add_argument('-Np', type=int, help='Number of peaks used in each spectral function')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs used at training')
    parser.add_argument('--server', action='store_true', help='Flag that states that we are running on server')
    parser.add_argument('--seed', type=int, default=916650397, help='Seed used in the calculation')

    # Parse the command line arguments
    return parser.parse_args()

if __name__ == '__main__':

    # Generate a parser to take command line arguments
    args = parse_arguments()

    # Load matplotlib style sheets if not on the server
    if not args.server:
        plt.style.use(['science', 'ieee', 'monospace'])

    # Set the seed to be constant
    torch.manual_seed(args.seed)

    # Generate the parameters
    param_A = Parameter('A', 0.1000, 1.0000)
    param_W = Parameter('W', 0.0500, 0.1000)
    param_M = Parameter('M', 0.1000, 3.5000)

    # Generate the gaussian ansatzs
    gauss = GaussianAnsatz(param_A, param_M, param_W)

    # Generate the kernel
    kernel = NRQCDKernel(64, 1000, [0.0, 8.0])

    # Generate the factory
    dataset = SPFactory([gauss for _ in range(args.Np)], kernel)

    # Generate the data: Increased from 64 to 128, check performance
    dataset.generate_data(args.Nb, args.Ns, use_GPU=True)

    # Wrap the dataset around a loader function
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # Generate a model to train
    model = ResidualNet(dataset.Nt, dataset.Ns, 'ResNet').cuda()

    # Optimiser and learning rate scheduler
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=15, factor=0.5, min_lr=1e-5, cooldown=5)

    # Get some random examples
    ex = torch.randint(0, dataset.Nb, (4,))

    # Save all epoch mean losses
    epoch_mean_losses = []

    # Folder where the run data will be stored
    run_path = Path(f'./runs/{model.name}/p{args.Np}_s{args.Ns}_b{args.Nb}')

    # Make the path if it does not exist
    run_path.mkdir(parents=True, exist_ok=True)

    # Train the model for different number of epochs
    for epoch in range(args.epochs):

        # Make the model trainable
        model.train()

        # Track the total loss and save the initial time
        epoch_loss, start = [], time.time()

        # Iterate through all minibatches
        for nb, (C_data, L_data) in enumerate(loader):

            # Set the gradients to zero
            optim.zero_grad()

            # Normalise the input and label data and move to the GPU
            C_data, L_data = C_data.cuda().log(), L_data.cuda()

            # Compute the prediction of the network
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

        print(
            f'Epoch {epoch + 1}: loss={epoch_loss.mean():.6f}, '
            f'lr={lr:.6f}, eta={time.time() - start} '
            f'-- {run_path.name}',
            flush=True
        )

        # Append the epoch mean losses
        epoch_mean_losses += [epoch_loss.mean()]

        if (epoch + 1) % 10 == 0:

            # Generate the prediction of the model
            with torch.no_grad():

                # Set the network in evaluation mode
                model.eval()

                # Generate a figure to plot the data
                fig = plt.figure(figsize=(10, 8))

                # Add several axis to the figure
                axis = [fig.add_subplot(ex.shape[0], 2, i) for i in range(1, 2 * ex.shape[0] + 1)]
                for ax in axis: ax.grid('#fefefe', alpha=0.6)

                # Get the label coefficients and the correlation functions
                L_data, C_data = dataset.L[ex, :], dataset.C[ex, :]

                # Compute the predicted coefficients from the scaled correlation functions
                L_pred = model(C_data.cuda().log()).cpu()

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

                # Make a folder to store the figures
                (run_path / 'figures').mkdir(parents=True, exist_ok=True)
                fig.savefig(run_path / 'figures' / f'epoch_{epoch + 1}.pdf')

                # Close the figures
                plt.cla(), plt.clf(), plt.close(fig)

    # Save the epoch mean loss and the model parameters in a file
    param_path = (run_path / 'params').mkdir(parents=True, exist_ok=True)
    torch.save({'loss': epoch_mean_losses, 'model': model.state_dict()}, run_path / 'params' / 'params.pt')

    # Plot the mean losses into a figure
    fig  = plt.figure(figsize=(6, 4))
    axis = fig.add_subplot()

    # Set some properties in the axis
    axis.set(xlabel='epoch', ylabel='loss')
    axis.grid('#fefefe', alpha=0.6)
    axis.plot(epoch_mean_losses, color='navy')
    fig.savefig(run_path / 'params' / 'loss.pdf')
