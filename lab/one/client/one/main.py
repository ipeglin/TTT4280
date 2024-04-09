import numpy as np
from os import path
import matplotlib.pyplot as plt
import sys

def raspi_import(path, channels=5):
  with open(path, 'r') as fid:
    sample_period = np.fromfile(fid, count=1, dtype=float)[0]
    data = np.fromfile(fid, dtype='uint16').astype('float64')
    data = data.reshape((-1, channels))

  sample_period *= 1e-6
  return sample_period, data

def format_data(data, num_bits=4096, supply_voltage=3.3):
  data /= num_bits
  data *= supply_voltage
  data -= supply_voltage / 2
  return data

if __name__ == "__main__":
  plt.rcParams.update({'font.size': 12})

  dirname, filename = path.split(path.abspath(__file__))
  sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1
  # else f'{dirname}/data/out-2024-01-16-10.09.25.bin')
  else f'/{dirname}/data/out-2024-02-14-08.43.52.bin')

  num_omitted_samples = 5000
  n_bits, ref_voltage = 12, 3.3
  data = format_data(data, num_bits=2**n_bits, supply_voltage=ref_voltage)
  data = data[:][num_omitted_samples:]

  samples_plotted = 500
  N, signal = data[:, 0].shape[0], data[:, 0]
  signal2 = data[:, 1]; signal3 = data[:, 2]
  signal4 = data[:, 3]; signal5 = data[:, 4]

  fig, axs = plt.subplots(5, 1, sharex=True, sharey=True)
  axs[0].plot(signal[:samples_plotted]); axs[0].set_title('Signal 1')
  axs[1].plot(signal2[:samples_plotted]); axs[1].set_title('Signal 2')
  axs[2].plot(signal3[:samples_plotted]); axs[2].set_title('Signal 3')
  axs[3].plot(signal4[:samples_plotted]); axs[3].set_title('Signal 4')
  axs[4].plot(signal5[:samples_plotted]); axs[4].set_title('Signal 5')
  fig.supxlabel('Tid (s)'); fig.supylabel('Amplitude (V)')
  fig.suptitle('Inngangssignaler')
  plt.tight_layout(); plt.show()

  # X = np.fft.fft(signal); X = np.fft.fftshift(X)
  # axis = np.arange(-N/2, N/2)
  # Sx = 20*np.log10(np.abs(X) / np.max(np.abs(X)))
  # plt.xlabel('Frekvens (Hz)'); plt.ylabel('Relativ effekt (dB)')
  # plt.title('Relativ effekt uten vindu, N='+str(N))
  # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
  # plt.plot(axis, Sx); plt.tight_layout(); plt.show()

  # pad_factor = 5
  # N *= pad_factor 
  # X = np.fft.fft(signal, N); X = np.fft.fftshift(X)
  # axis = np.arange(-N/2, N/2)
  # Sx = 20*np.log10(np.abs(X) / np.max(np.abs(X)))
  # plt.xlabel('Frekvens (Hz)'); plt.ylabel('Relativ effekt (dB)')
  # plt.title('Relativ effekt med padding, N='+str(N))
  # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
  # plt.plot(axis, Sx); plt.tight_layout(); plt.show()

  # hanning = np.hanning(len(signal))
  # signal *= hanning

  # X = np.fft.fft(signal, N); X = np.fft.fftshift(X)
  # Sx = 20*np.log10(np.abs(X) / np.max(np.abs(X)))
  # plt.xlabel('Frekvens (Hz)'); plt.ylabel('Relativ effekt (dB)')
  # plt.title('Relativ effekt med Hann-vindu, N='+str(N))
  # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
  # plt.plot(axis, Sx); plt.tight_layout(); plt.show()

  # delta = ref_voltage / (2**n_bits)
  # snr = 10 * np.log10((12 * (2 ** n_bits * delta) ** 2) / (8 * delta) ** 2)
  # print(f'SNR = {snr:.2f} dB')