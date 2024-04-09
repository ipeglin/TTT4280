from os import listdir, path
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

dirname, filename = path.split(path.abspath(__file__))
sets = ['refl']

results = np.empty((0, 3))

for measurement_set in sets:
  files = listdir(f'{dirname}/data/{measurement_set}')
  print(f'''{'#' * (27 + len(str(len(files))) + len(measurement_set))}
## Found {len(files)} files for set {measurement_set} ##
{'#' * (27 + len(str(len(files))) + len(measurement_set))}
''')
    
  set_result = np.empty((0, 3))

  for file in files:
    print('File:', file)
    with open(f'{dirname}/data/{measurement_set}/{file}') as f:
        data = np.empty((0, 3)) # RGB
        lines = f.readlines()

        for i, line in enumerate(lines):
          rgb = np.array([line.split(' ')]).astype(float)
          data = np.append(data, rgb, axis=0)
        
    # remove first N samples
    num_samples_omitted = 2
    data = data[:][num_samples_omitted:]
    total_sampling_time = 30 #s
    num_samples = data.shape[0]

    sampling_freq = num_samples / total_sampling_time
    sampling_period = 1 / sampling_freq

    # print(f'''
    # Sampling frequency: {sampling_freq} Hz
    # Sampling period: {sampling_period * 10 ** 3:.3f} ms
    # ''')

    red = data[:, 0]
    green = data[:, 1]
    blue = data[:, 2]

    red_avg = np.mean(red)
    green_avg = np.mean(green)
    blue_avg = np.mean(blue)

    red_std = np.std(red)
    green_std = np.std(green)
    blue_std = np.std(blue)

    # print(f'''
    # Red: Avg[r] = {red_avg}, σ_r = {red_std}
    # Green: Avg[g] = {green_avg}, σ_g = {green_std}
    # Blue: Avg[b] = {blue_avg}, σ_b = {blue_std}
    # ''')

    # Plot channels in time in subfigures
    fig, axs = plt.subplots(3, sharex=True)
    axis = np.arange(0, total_sampling_time, sampling_period)
    fig.suptitle(f'RGB channels in time ({file})')
    fig.supxlabel('Time (s)')
    axs[0].plot(axis, red, color='r')
    axs[0].set_title('Red')
    axs[1].plot(axis, green, color='g')
    axs[1].set_title('Green')
    axs[2].plot(axis, blue, color='b')
    axs[2].set_title('Blue')

    plt.tight_layout()

    # Detrend signals
    filter_order = 2
    bpf_freqs = (40 / 60, 230 / 60)
    bpf = signal.butter(filter_order, bpf_freqs, fs=sampling_freq, btype='bandpass', output='sos')

    red_filtered = signal.sosfilt(bpf, red)
    green_filtered = signal.sosfilt(bpf, green)
    blue_filtered = signal.sosfilt(bpf, blue)

    axis = np.arange(0, total_sampling_time, sampling_period)

    # Plot filtered channels in time in subfigures
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(f'RGB freq components in time, $f \in [{ "{:.3f}, {:.3f}".format(*bpf_freqs) }]$ ({file})')
    fig.supxlabel('Time (s)')
    axs[0].plot(axis, red_filtered, color='r')
    axs[0].set_title('Red')
    axs[1].plot(axis, green_filtered, color='g')
    axs[1].set_title('Green')
    axs[2].plot(axis, blue_filtered, color='b')
    axs[2].set_title('Blue')

    plt.tight_layout()

    # Take FFT of each channel
    num_samples *= 15
    df = sampling_freq/num_samples
    fft_ax = np.arange(-sampling_freq/2, sampling_freq/2, df)
    red_fft = np.abs(np.fft.fft(red_filtered, num_samples))
    red_fft = np.fft.fftshift(red_fft)
    green_fft = np.abs(np.fft.fft(green_filtered, num_samples))
    green_fft = np.fft.fftshift(green_fft)
    blue_fft = np.abs(np.fft.fft(blue_filtered, num_samples))
    blue_fft = np.fft.fftshift(blue_fft)

    red_freq_max_at = np.argmax(np.abs(red_fft))
    green_freq_max_at = np.argmax(np.abs(green_fft))
    blue_freq_max_at = np.argmax(np.abs(blue_fft))

    red_freq_max = np.abs(fft_ax[red_freq_max_at])
    green_freq_max = np.abs(fft_ax[green_freq_max_at])
    blue_freq_max = np.abs(fft_ax[blue_freq_max_at])

    print(f'''
    Red Freq={red_freq_max:.3f} Hz
    Green Freq={green_freq_max:.3f} Hz
    Blue Freq={blue_freq_max:.3f} Hz
    ''')

    red_pulse = 60 * red_freq_max
    green_pulse = 60 * green_freq_max
    blue_pulse = 60 * blue_freq_max

    print(f'''
    Red HR={red_pulse:.3f} BPM
    Green HR={green_pulse:.3f} BPM
    Blue HR={blue_pulse:.3f} BPM
    ''')

    set_result = np.append(set_result, [[red_pulse, green_pulse, blue_pulse]], axis=0)

    # Plot FFTs
    fig, axs = plt.subplots(3)
    fig.suptitle(f'FFT of RGB channels ({file})')
    axs[0].plot(fft_ax, np.abs(red_fft), color='r')
    axs[0].set_title('Red')
    axs[1].plot(fft_ax, np.abs(green_fft), color='g')
    axs[1].set_title('Green')
    axs[2].plot(fft_ax, np.abs(blue_fft), color='b')
    axs[2].set_title('Blue')

    axs[0].set_xlim(bpf_freqs)
    axs[1].set_xlim(bpf_freqs)
    axs[2].set_xlim(bpf_freqs)
    # axs[0].set_xlim(-bpf_freqs[1],bpf_freqs[1])
    # axs[1].set_xlim(-bpf_freqs[1],bpf_freqs[1])
    # axs[2].set_xlim(-bpf_freqs[1],bpf_freqs[1])

    plt.tight_layout()

    snr_bpm_threshold = 5 # bpm
    snr_freq_threshold = snr_bpm_threshold / 60
    snr_index_threshold = int(np.round(snr_freq_threshold / df))
    # snr_index_threshold = 50


    red_signal_interval = (red_freq_max_at - snr_index_threshold, red_freq_max_at + snr_index_threshold + 1)
    green_signal_interval = (green_freq_max_at - snr_index_threshold, green_freq_max_at + snr_index_threshold + 1)
    blue_signal_interval = (blue_freq_max_at - snr_index_threshold, blue_freq_max_at + snr_index_threshold + 1)

    # print intervals
    print(f'''
    Red signal interval: {red_signal_interval}
    Green signal interval: {green_signal_interval}
    Blue signal interval: {blue_signal_interval}
    ''')

    red_signal = np.mean(red_fft[red_signal_interval[0]:red_signal_interval[1]])
    green_signal = np.mean(green_fft[green_signal_interval[0]:green_signal_interval[1]])
    blue_signal = np.mean(blue_fft[blue_signal_interval[0]:blue_signal_interval[1]])

    red_noise_mask = np.ones(len(red_fft), dtype=bool)
    red_noise_mask[red_signal_interval[0]:red_signal_interval[1]] = False
    red_noice = np.mean(red_fft[red_noise_mask])

    green_noise_mask = np.ones(len(green_fft), dtype=bool)
    green_noise_mask[green_signal_interval[0]:green_signal_interval[1]] = False
    green_noice = np.mean(green_fft[green_noise_mask])

    blue_noise_mask = np.ones(len(blue_fft), dtype=bool)
    blue_noise_mask[blue_signal_interval[0]:blue_signal_interval[1]] = False
    blue_noice = np.mean(blue_fft[blue_noise_mask])



    print(f'''
    SNR (Red): {red_signal / red_noice}
    SNR (Green): {green_signal / green_noice}
    SNR (Blue): {blue_signal / blue_noice}
    ''')
    


    plt.show()


  avg_pulse_red = np.mean(set_result[:, 0])
  avg_pulse_green = np.mean(set_result[:, 1])
  avg_pulse_blue = np.mean(set_result[:, 2])

  std_pulse_red = np.std(set_result[:, 0])
  std_pulse_green = np.std(set_result[:, 1])
  std_pulse_blue = np.std(set_result[:, 2])
  print(f'''Channel measurements:
  R: {set_result[:, 0]}
  G: {set_result[:, 1]}
  B: {set_result[:, 2]}
''')

  print(f'''
{'#' * (22 + len(measurement_set))}
## Results for set {measurement_set} ##
{'#' * (22 + len(measurement_set))}
Avg Pulse (Red): {avg_pulse_red} BMP, σ_r: {std_pulse_red}
Avg Pulse (Green): {avg_pulse_green} BMP, σ_g: {std_pulse_green}
Avg Pulse (Blue): {avg_pulse_blue} BMP, σ_b: {std_pulse_blue}
''')
