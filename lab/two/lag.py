import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate



def w(mean, std_dev):
  return np.random.normal(mean, std_dev)

def f(x):
  return amplitude * np.sin(2 * np.pi * freq * x) + w(0, 0.25)

def lag_signal(signal, N=0):
  if N > 0: # Positive delay
    signal2 = np.zeros_like(signal)
    signal2[N:] = signal[:-N]
    signal2[:N] = 0 # Pad the beginning with zeros
    
    return signal2

  if N < 0: # Negative delay
      signal2 = np.zeros_like(signal)
      signal2[:N] = signal[-N:]
      signal2[N:] = 0 # Pad the end with zeros

      return signal2

  # No delay
  return signal.copy()

def get_signal_lag(signal, signal2):
  len1 = len(signal)
  len2 = len(signal2)

  if len1 != len2:
    raise ValueError('Signal lengths do not match')
  
  x = np.linspace(-(len1 - 1), len1 - 1, 2 * len1 - 1)
  corr =  correlate(signal, signal2, mode='full')
  corr = np.abs(corr)

  return x[np.argmax(corr)]

if __name__ == '__main__':
  amplitude =  1
  freq = 100
  dt = 0.25e-3
  num_samples = 50
  fs = 1 / dt

  x = np.arange(0, num_samples)
  # signal = f(x)
  signal = np.random.randint(5, size=num_samples)

  # offset by N samples
  N = 10

  if N > 0: # Positive delay
      signal2 = np.zeros_like(signal)
      signal2[N:] = signal[:-N]
      signal2[:N] = 0 # Pad the beginning with zeros

  elif N < 0: # Negative delay
      signal2 = np.zeros_like(signal)
      signal2[:N] = signal[-N:]
      signal2[N:] = 0 # Pad the end with zeros

  else: # No delay
      signal2 = signal.copy()

  x = np.linspace(-(num_samples - 1), num_samples - 1, 2 * num_samples - 1)
  corr =  correlate(signal, signal2, mode='full')
  corr = np.abs(corr)

  print('Max lag:', x[np.argmax(corr)])

  plt.title('Correlation')
  plt.xlabel('Lag (l)')
  plt.ylabel('Amplitude')
  plt.stem(x, corr)
  plt.show()