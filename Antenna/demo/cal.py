import matplotlib.pyplot as plt
import numpy as np

import array_index
import aoa
import aoa_1d


def cal(A: list, channel):
    I = A[0::2]
    Q = A[1::2]
    S = [complex(I[i], -Q[i]) for i in range(len(I))]
    d = np.diff(np.angle(S))
    d = np.mod(d, 2 * np.pi)
    dphase = np.mean(d[0:7])
    # print(dphase)
    phaseS = np.angle(S)
    for i in range(8):
        phaseS[i] = np.mod(phaseS[i] - dphase * i, 2 * np.pi)
    for i in range(8, len(phaseS)):
        phaseS[i] = np.mod(phaseS[i] - dphase * 7 - dphase * (i - 7) * 2, 2 * np.pi)
    phaseS = array_index.rindex_array(phaseS[7:])
    # print(phaseS)
    # plt.plot(phaseS[0])
    # plt.show()
    music_worker = aoa_1d.AOA_1D()
    phase = -phaseS[0, 1:17]
    phase = phase.reshape((16, 1))
    signal = np.exp(1j * phase)
    # psp, azimuth, elevation = music_worker.music(signal)
    # print(azimuth * 180 / np.pi, elevation * 180 / np.pi)

    psp, azimuth, intensity = music_worker.music(signal,channel)
    # print(azimuth * 180 / np.pi)

    # psp, azimuth, intensity = music_worker.p0(phase)
    # print(azimuth)
    # print(phase.reshape(1, -1)[0])
    print(psp.shape)
    return phase.reshape(1, -1)[0], psp
    # phaseS[0, 2] - phaseS[0, 1]
