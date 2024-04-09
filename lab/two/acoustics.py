import numpy as np
from os import listdir, path
from mic_array import TriangleArray
from data_handler import raspi_import, format_data, strip_data
from lag import get_signal_lag
    
if __name__ == '__main__':
  plotted_lags = False
  f_s = 31250
  c = 343
  d = 0.065
  max_lags = np.ceil(f_s * d / c).astype(int)
  
  print('Max lags:', max_lags)

  # define sensor positions
  dist_to_origo = 0.01 # metres
  sensor_1 = np.array([0, 1])
  sensor_2  = np.array([-np.sqrt(3)/2, -0.5])
  sensor_3 = np.array([np.sqrt(3)/2, -0.5])

  # define sensor array
  array = TriangleArray(sensor_1, sensor_2, sensor_3, dist_to_origo)

  # import data
  dirname, filename = path.split(path.abspath(__file__))
  sets = ['kl12', 'kl14', 'kl16', 'kl19']

  results = np.array([])

  for measurement_set in sets:
    files = listdir(f'{dirname}/data/{measurement_set}')
    print(f'''########################################
## Found {len(files)} files for set {measurement_set} 
########################################
''')

    tmp = []

    for file in files:
      sample_period, data = raspi_import(f'{dirname}/data/{measurement_set}/{file}')
      data = format_data(data)
      print(data, data.shape)

      # extract acoustic signals
      signal_1 = data[:, 3][1000:]
      signal_2 = data[:, 2][1000:]
      signal_3 = data[:, 4][1000:]

      if not plotted_lags:
        import matplotlib.pyplot as plt
        from scipy.signal import correlate

        length = len(signal_1)

        # plot lags
        fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
        x = np.arange(-max_lags, max_lags + 1)
        axs[0].plot(x, correlate(signal_1, signal_1)[length-1-max_lags:length+max_lags])
        axs[0].set_title('Kysskorrelasjon 1')
        
        axs[1].plot(x, correlate(signal_2, signal_1)[length-1-max_lags:length+max_lags])
        axs[1].set_title('Korrelasjon 21')

        axs[2].plot(x, correlate(signal_3, signal_1)[length-1-max_lags:length+max_lags])
        axs[2].set_title('Korrelasjon 31')
        
        axs[3].plot(x, correlate(signal_3, signal_2)[length-1-max_lags:length+max_lags])
        axs[3].set_title('Korrelasjon 32')


        plt.suptitle('Korrelasjoner mellom mikrofonsignalene')
        plt.tight_layout()

        plotted_lags = True

      # max lags
      tau_21 = get_signal_lag(signal_2, signal_1)
      tau_31 = get_signal_lag(signal_3, signal_1)
      tau_32 = get_signal_lag(signal_3, signal_2)

      
      delays = np.array([tau_21, tau_31, tau_32])
      theta = array.compute_angle(delays, verbose=True)

      tmp.append(np.degrees(theta))

    results = np.append(results, np.mean(tmp))

print('Results:', results)

if plotted_lags:
  plt.show()




      