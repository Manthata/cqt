import torch
from nnAudio.Spectrogram import CQT1992v2
import matplotlib.pylab as plt
import numpy as np

# sr is the sampling rate, it is 2048 Hz
# fmax is half the sampling rate

cqt_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64)

def run_cqt_transform(x: np.array) -> torch.Tensor:
    # We stack the passed x since there are 3
    # time series per file.
    x = np.hstack(x)
    # Normalize (is there a better way?)
    x = x / np.max(x)
    x = torch.from_numpy(x).float()
    return cqt_transform(x)

# Running on one file and plotting the result.
x = np.load("path/to/file.npy")

# We take the first (and only) result since the 
# result is batch-shaped ((1, freq_bins, time_steps)).
img = run_cqt_transform(x)[0]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img)
plt.show()
