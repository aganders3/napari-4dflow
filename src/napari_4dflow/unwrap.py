"""
Phase unwrapping algorithms.

References:
----------
Loecher M, Schrauben E, Johnson KM, Wieben O. Phase unwrapping in 4D MR flo
with a 4D single-step laplacian algorithm. J Magn Reson Imaging. 201
Apr;43(4):833-42. doi: 10.1002/jmri.25045. Epub 2015 Sep 28. PMID: 26417641.
"""

from functools import partial

import numpy as np


def unwrap(phase, axes=None):
    """Unwrap a phase array using N-D Laplacian.

    Args:
        phase: wrapped input array (-pi to pi)
        axes: axes to analyze for phase wrapping (experimental)

    Returns:
        nr: integer array containing the NUMBER of wraps per voxel
            note that this is not the actual unwrapped data
            use phase += 2*pi*nr to unwrap
    """
    if axes is None:
        axes = list(range(phase.ndim))
    unwrap_shapes = [
        phase.shape[a] if a in axes else 1 for a in range(phase.ndim)
    ]

    laplace_kernel_freq = np.meshgrid(
        *[np.arange(s, dtype=np.float32) - s // 2 for s in unwrap_shapes],
        indexing="ij",
    )

    laplace_kernel_freq = np.sum(
        2 * np.cos(np.pi * d / s)
        for d, s in zip(laplace_kernel_freq, unwrap_shapes)
    ) - 2 * len(unwrap_shapes)
    # TODO: scale dimensions by voxel size / time step

    laplace = partial(_apply_freq_kernel, kernel=laplace_kernel_freq)

    phase_laplace_wrapped = laplace(phase)
    phase_laplace_no_wrap = np.cos(phase) * laplace(np.sin(phase)) - np.sin(
        phase
    ) * laplace(np.cos(phase))
    phase_diff = laplace(
        phase_laplace_no_wrap - phase_laplace_wrapped,
        reverse=True,
    )
    nr = np.rint(phase_diff / 2 / np.pi).astype(np.int8)
    return nr


def _apply_freq_kernel(data, kernel, *, reverse=False):
    """Apply a frequency domain kernel to data."""
    if reverse:
        # prevent divide-by-zero for reverse kernel
        kernel = kernel.copy()
        kernel[kernel == 0] = kernel.min()
        kernel = 1 / kernel

    data_freq = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data)))

    return np.fft.fftshift(
        np.real(np.fft.ifftn(np.fft.ifftshift(data_freq * kernel)))
    )
