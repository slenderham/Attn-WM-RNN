from matplotlib import pyplot as plt
from utils import load_checkpoint
from models import LeakyRNN
import torch
import os

def plot_connectivity(x2hw, h2hw, h2ow):
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(h2hw, cmap='seismic')
    axes[0, 1].imshow(h2ow.T, cmap='seismic')
    axes[1, 0].imshow(x2hw, cmap='seismic')
    fig.colorbar()
    plt.show()

def plot_training_curve(l):
    fig, ax = plt.subplots()
    ax.plot(l)
    ax.set_xlabel('Training iterations')
    ax.set_xlabel('Loss')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Directory of trained model')
    parser.add_argument('--connectivity', action='store_true')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--training_curve', action='store_true')
    args = parser.parse_args()

    # TODO: add dict for model params and save it
    state_dict = torch.load(os.path.join(args.exp_dir, 'checkpoint.pth.tar'), map_location=torch.device('cpu'))
    metrics = open(os.path.join(args.exp_dir, 'metrics.txt')).readlines()
    metrics = [float(m) for m in metrics]

    if args.connectivity:
        plot_connectivity(state_dict['x2h.weight'], state_dict['h2h.weight'], state_dict['h2o.weight'])
    if args.training_curve:
        plot_training_curve()