import torch
import numpy as np
from scipy.fftpack import fft2, fftshift, ifftshift, ifft2

class AngularForward:

    def __init__(self, angf, freq, c, h, distance):
        self.angf = angf
        self.freq = freq
        self.c = c
        self.h = h
        self.distance = distance

    def process(self, hologram):
        f0 = self.freq  # frequency
        cwater = self.c  # m/s, sound speed in water
        k0 = 2 * np.pi * f0 / cwater  # wave vector

        I1 = hologram.double()
        nx, ny = I1.shape  # pixel numbers of the hologram
        # X, Y = np.meshgrid(np.linspace(-(nx / 2 - 1), nx / 2, nx) * self.h / nx)


        dx = self.h / nx
        dy = self.h / ny

        nx1 = self.angf  # number of angular frequency modes
        ny1 = self.angf  # number of angular frequency modes

        dkx1 = 2 * np.pi / dx / (2 * nx1 - 1)
        dky1 = 2 * np.pi / dy / (2 * ny1 - 1)

        kex1 = np.arange(-nx1 + 1, nx1) * dkx1
        key1 = np.arange(-ny1 + 1, ny1) * dky1
        Kx1, Ky1 = np.meshgrid(kex1, key1)

        fftI1 = fft2(I1.numpy()[:nx, :ny], [2 * nx1 - 1, 2 * ny1 - 1])
        fftI2 = fftshift(fftI1)

        H = torch.tensor(np.exp(1j * (self.distance) * np.sqrt(np.maximum(k0 ** 2 - Kx1 ** 2 - Ky1 ** 2, 0))), dtype=torch.cdouble)


        mask = Kx1 ** 2 + Ky1 ** 2 > k0 ** 2
        H[mask] = 0

        fftI3 = torch.from_numpy(fftI2) * H

        forward_propagated_fft = ifftshift(fftI3.numpy())
        forward_propagated_image = ifft2(forward_propagated_fft, [2 * nx1 - 1, 2 * ny1 - 1])

        image = torch.abs(torch.from_numpy(forward_propagated_image[:nx, :ny]))
        image = image / image.max()

        return image
