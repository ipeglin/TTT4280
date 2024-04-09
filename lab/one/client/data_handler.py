import numpy as np
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

def get_sensor_data(data, n=0):
  if n < 0:
    return data[:, 0]
  
  return data[:, n]

if __name__ == "__main__":
  sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1
  else '/Users/ipeglin/Downloads/out-2024-01-13-16.45.32.bin')
  
  data = format_data(data)
  
  # 5 subplots
  fig, axs = plt.subplots(5, 1)

  N, signal = data[:, 0].shape[0], data[:, 0]
  axs[0].plot(np.arange(data.shape[0])*sample_period, signal)
  
  signal2 = data[:, 1]
  axs[1].plot(np.arange(data.shape[0])*sample_period, signal2)
  
  signal3 = data[:, 2]
  axs[2].plot(np.arange(data.shape[0])*sample_period, signal3)
  
  signal4 = data[:, 3]
  axs[3].plot(np.arange(data.shape[0])*sample_period, signal4)
  
  signal5 = data[:, 4]
  axs[4].plot(np.arange(data.shape[0])*sample_period, signal5)

  plt.xlabel('Tid (s)'); plt.ylabel('Tall fra ADC-en')

  plt.show()

  # fft, fft_x = np.fft.fft(signal), np.fft.fftfreq(N, sample_period)
  # plt.plot(fft_x, np.abs(fft))
  # plt.title('FFT uten vindu, N='+str(N)); plt.xlim(-50, 50); plt.xlabel('Frekvens (Hz)')
  # plt.show()

  # N *= 5
  # padded = np.pad(signal, (0, N - len(signal)), mode='constant', constant_values=0)
  # padded_len = len(padded)
  
  # fft, fft_x = np.fft.fft(padded), np.fft.fftfreq(N, sample_period)
  # plt.plot(fft_x, np.abs(fft))
  # plt.title('FFT uten vindu, N='+str(N))
  # plt.xlim(-50, 50)
  # plt.xlabel('Frekvens (Hz)')
  # plt.show()
  
  # hanning = np.hanning(padded_len)
  # padded *= hanning
  
  # fft = np.fft.fft(padded)
  # fft_x = np.fft.fftfreq(N, sample_period)
  # plt.plot(fft_x, np.abs(fft))
  # plt.title('FFT med hanning-vindu')
  # plt.xlim(-50, 50)
  # plt.xlabel('Frekvens (Hz)')
  # plt.show()