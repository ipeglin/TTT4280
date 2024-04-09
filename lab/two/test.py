# c = 343 # speed of sound in air

  # # defined sensor positions
  # dist_to_origo = 0.01 # metres
  # x_1 = np.array([0, 1]) * dist_to_origo
  # x_2 = np.array([- np.sqrt(3)/2, -0.5]) * dist_to_origo
  # x_3 = np.array([np.sqrt(3)/2, -0.5]) * dist_to_origo

  # print(f'''Sensor positions:
  #       Sensor 1: {x_1}
  #       Sensor 2: {x_2}
  #       Sensor 3: {x_3}
  # ''')

  # # distance vectors
  # x_21 = x_2 - x_1
  # x_31 = x_3 - x_1
  # x_32 = x_3 - x_2
  # distvec = np.array([x_21, x_31, x_32])

  # print(f'''Sensor distance vectors:
  #       x_21: {x_21}
  #       x_31: {x_31}
  #       x_32: {x_32}
  # ''')


  # # least squares solution
  # x_comps = distvec[:][:, 0]
  # y_comps = distvec[:][:, 1]

  # # computing coefficients
  # A = 2 * np.sum(x_comps ** 2)
  # B = 2 * np.sum(x_comps * y_comps) # Always zero in triangle configuration
  # D = 2 * np.sum(y_comps ** 2)

  # # Order: tau_21, tau_31, tau_32
  # # TODO: Implement correlation function with measurement data
  # delays = np.array([0, 2, 4])

  # # computing estimates
  # x = ((2 * c) / A) * np.sum(x_comps * delays)
  # y = - ((2 * c) / D) * np.sum(y_comps * delays)

  # print(f'''Estimates:
  #       x: {x}
  #       y: {y}
  # ''')

  # # 180 degree resolution
  # if x < 0:
  #   x += np.pi
  #   print('Compensating for x < 0')

  # # compute angle
  # try:
  #   theta = np.arctan(y / x)
  #   print(f'Incoming angle: {theta}')
  # except ZeroDivisionError:
  #   print('Incoming angle: 90 degrees')