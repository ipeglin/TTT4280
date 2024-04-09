import numpy as np
from data_handler import raspi_import, format_data, strip_data
from os import listdir, path, makedirs
import matplotlib.pyplot as plt
import scipy.signal as signal

SHOW_PLOT = False
SAVE_PLOT = True

if __name__ == '__main__':
  figure_font_size = 12
  font = {
    'size': figure_font_size
  }
  plt.rc('font', **font)

  dirname, filename = path.split(path.abspath(__file__))
  sets = ['forward1', 'forward2', 'reverse']

  results = {}
  
  for measurement_set in sets:
    files = listdir(f'{dirname}/data/{measurement_set}')
    print(f'''{'#' * (27 + len(str(len(files))) + len(measurement_set))}
## Found {len(files)} files for set {measurement_set} ##
{'#' * (27 + len(str(len(files))) + len(measurement_set))}
''')
    
    results[measurement_set] = {
      'velocities': [],
      'snr': [],
    }
    
    if SAVE_PLOT:
      try:
        makedirs(f'{dirname}/plots/{measurement_set}')
      except FileExistsError:
        print("WARNING! Directory already exists")

    for file in files:
      if file == '.DS_Store':
        continue

      sample_period, data = raspi_import(f'{dirname}/data/{measurement_set}/{file}')
      data = format_data(data)

      valid_samples_starts_at = {
        'forward1': 8500,
        'forward2': 11500,
        'reverse': 12000,
      }

      try:
        omitted_samples = valid_samples_starts_at[measurement_set]
      except:
        omitted_samples = 0

      sign_i = signal.detrend(data[:, 0][omitted_samples:])
      sign_q = signal.detrend(data[:, 1][omitted_samples:])

      sign = sign_i + 1j * sign_q
      sign = sign

      sample_freq = 1 / sample_period
      num_samples = data.shape[0] - omitted_samples

      # Plot channels in time in subfigures
      axis = np.arange(num_samples) * sample_period
      fig, ax = plt.subplots()
      fig.supxlabel('Tid (s)')
      fig.supylabel('Amplitude (V)')
      fig.suptitle(f'I-Q kanaler v. tid ({file})')
      ax.plot(axis, np.vstack([sign.real, sign.imag]).T)
      plt.tight_layout()

      if SAVE_PLOT:
        plt.savefig(f'{dirname}/plots/{measurement_set}/{file}_time.svg')

      num_samples *= 15
      df = sample_freq/num_samples
      fft_ax = np.arange(-sample_freq/2, sample_freq/2, df)
      fft = np.fft.fft(sign, num_samples)
      fft = np.fft.fftshift(fft)
      rel_fft = 20*np.log10(fft / np.max(fft))
      freq_max_at = np.argmax(fft)

      freq_max = fft_ax[freq_max_at]
      doppler_freq = freq_max
      print(f'Doppler freqency: {doppler_freq:.3f} Hz')

      # compute velocity
      center_freq = 24.13 * (10 ** 9)
      c = 3 * (10 ** 8)
      
      velocity = doppler_freq * c / (2 * center_freq)
      results[measurement_set]['velocities'].append(velocity)

      # Plot FFT
      fig, ax = plt.subplots()
      fig.supxlabel('Frekvens (Hz)')
      fig.suptitle(f'FFT av I-Q kanaler ({file})')
      ax.annotate(f"{freq_max:.3f} Hz", (freq_max + 10000*df, np.max(rel_fft) - 0.75*figure_font_size)) # Offset annotation text
      ax.plot(freq_max, np.max(rel_fft), marker="o", color="black")
      ax.plot(fft_ax, rel_fft, color='b')
      plt.tight_layout()

      if SAVE_PLOT:
        plt.savefig(f'{dirname}/plots/{measurement_set}/{file}_fft.svg')

      car_top_speed = 3 # m/s
      car_max_doppler_freq = int(2 * car_top_speed * center_freq / c)

      zero_freq_index = np.where(fft == np.max(fft))[0][0]
      signal_interval = (zero_freq_index - car_max_doppler_freq, zero_freq_index + car_max_doppler_freq + 1)

      sign_s = np.mean(np.abs(fft[signal_interval[0]:signal_interval[1]]))
      
      noise_mask = np.ones(len(fft), dtype=bool)
      noise_mask[signal_interval[0]:signal_interval[1]] = False
      noice = np.mean(np.abs(fft[noise_mask]))

      snr = sign_s / noice
      results[measurement_set]['snr'].append(snr)
      print(f'SNR (Signal): {snr}')
      
    results[measurement_set]['std_vel'] = np.std(results[measurement_set]['velocities'])
    results[measurement_set]['std_snr'] = np.std(results[measurement_set]['snr'])
    
    if SHOW_PLOT:
      plt.show()
      
  print('Results:', results)