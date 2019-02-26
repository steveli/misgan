import numpy as np
import pylab as plt
from matplotlib.patches import Rectangle
from PIL import Image


def plot_grid(image, bbox=None, gap=0, gap_value=1,
              nrow=4, ncol=8, save_file=None):
    image = image.cpu().numpy()
    channels, len0, len1 = image[0].shape
    grid = np.empty(
        (nrow * (len0 + gap) - gap, ncol * (len1 + gap) - gap, channels))
    # Convert to W, H, C
    image = image.transpose((0, 2, 3, 1))
    grid.fill(gap_value)

    for i, x in enumerate(image):
        if i >= nrow * ncol:
            break
        p0 = (i // ncol) * (len0 + gap)
        p1 = (i % ncol) * (len1 + gap)
        grid[p0:(p0 + len0), p1:(p1 + len1)] = x

    # figsize = np.r_[ncol, nrow] * .75
    scale = 2.5
    figsize = ncol * scale, nrow * scale   # FIXME: scale by len0, len1
    fig = plt.figure(figsize=figsize)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    grid = grid.squeeze()
    ax.imshow(grid, cmap='binary_r', interpolation='none', aspect='equal')

    if bbox is not None:
        nplot = min(len(image), nrow * ncol)
        for i in range(nplot):
            if len(bbox) == 1:
                d0, d1, d0_len, d1_len = bbox[0]
            else:
                d0, d1, d0_len, d1_len = bbox[i]
            p0 = (i // ncol) * (len0 + gap)
            p1 = (i % ncol) * (len1 + gap)
            offset = np.array([p1 + d1, p0 + d0]) - .5
            ax.add_patch(Rectangle(
                offset, d1_len, d0_len, lw=4, edgecolor='red', fill=False))

    if save_file:
        fig.savefig(save_file)
    plt.close(fig)


def plot_samples(samples, save_file, nrow=4, ncol=8):
    x = samples.cpu().numpy()
    channels, len0, len1 = x[0].shape
    x_merge = np.zeros((nrow * len0, ncol * len1, channels))

    for i, x_ in enumerate(x):
        if i >= nrow * ncol:
            break
        p0 = (i // ncol) * len0
        p1 = (i % ncol) * len1
        x_merge[p0:(p0 + len0), p1:(p1 + len1)] = x_.transpose((1, 2, 0))

    x_merge = (x_merge * 255).clip(0, 255).astype(np.uint8)
    # squeeze() to remove the last dimension for the single-channel image.
    im = Image.fromarray(x_merge.squeeze())
    im.save(save_file)
