import numpy as np

def lag_signal(signal, l=None):
  if l is None:
    return signal

  temp = np.zeros_like(signal)  # Create an array of zeros with the same shape as the original signal

  if l > 0:
    temp[l:] = signal[:-l] 
  else:
    temp[:l] = signal[-l:]

  return temp

if __name__ == "__main__":
  signal = np.array([1, 2, 3, 4, 5])
  print(signal)
  print(lag_signal(signal, 1))