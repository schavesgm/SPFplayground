import re
from collections import deque
from pathlib import Path
from typing import Optional
from typing import Generator
from typing import Union

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from natsort import natsorted

# Define some types
IterDir = Generator[Path, None, None]
Int     = Optional[int]
Number  = Union[float, int]

def filter_runs(runs: IterDir, Nb: Int = None, Ns: Int = None, Np: Int = None) -> list[Path]:
    f = lambda x: re.match(f'p{Np if Np else ".*"}_s{Ns if Ns else ".*"}_b{Nb if Nb else ".*"}', x.name)
    return list(natsorted(filter(f, runs), lambda x: x.name))

def get_runparams(run: Path) -> list[str]:
    return [re.match('[bsp](\d+)', x).group(1) for x in run.name.split('_')]

def smooth_curve(data: list[Number], lags: int) -> list[Number]:
    queue, smoothed = deque(maxlen=lags), []
    for element in data:
        queue.append(element)
        smoothed.append(sum(queue) / len(queue))
    return smoothed

colors     = ["#333466", "#d65533", "#61a082", "#eeb861", "#ca7892", "#8cabea", "#572b9e", "#09c553", "#2e5b4a", "#b28bef"]

if __name__ == "__main__":

    plt.style.use(['science', 'ieee', 'monospace'])

    # Define the runs to be retrieved
    Nb: Int = 150000
    Ns: Int = None
    Np: Int = None

    # Filter the runs according to some values
    model_runs = filter_runs(Path('../runs/ResNet').iterdir(), Nb=Nb, Ns=Ns, Np=Np)

    # Eliminate all the runs whose params.pt does not exist
    model_runs = list(filter(lambda x: (x / 'params/params.pt').exists(), model_runs))

    # Figure where the losses will be plotted
    fig  = plt.figure(figsize=(6.5, 5.0))
    axis = fig.add_axes([0.05, 0.05, 0.90, 0.90])

    # Set some properties in the axis
    axis.set(xlabel='epoch', ylabel='loss', yscale='log')
    axis.grid('#fefefe', alpha=0.6)

    for i, run in enumerate(model_runs):

        params = torch.load(run / 'params/params.pt')
        rNp, rNs, rNb = get_runparams(run)

        # Generate the label for the result
        label = f"N_b = {rNb}," if not Nb else "" \
                f"N_p = {rNp}," if not Np else "" \
                f"N_s = {rNs}"  if not Ns else ""

        # Trim the last comma from the label
        label = label[:-1] if label.endswith(',') else label

        # Plot the data in the figure
        curve = smooth_curve(params['loss'], 10)
        axis.plot(curve, label=f'${label}$' if label else None, color=colors[i], lw=2)

    axis.set_ylim(top=1.0)
    axis.legend(frameon=False)
    
    path_to_figures = Path('../runs/figures/ResNet/')
    path_to_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_to_figures / 'losses.pdf')
