import numpy as np

class TriangleArray():
  c = 343 # speed of sound in air
  def __init__(self, x_1=None, x_2=None, x_3=None, dist_to_origo=None):
    self.x_1 = x_1
    self.x_2 = x_2
    self.x_3 = x_3
    self.dist_to_origo = dist_to_origo

    x_21 = x_2 - x_1
    x_31 = x_3 - x_1
    x_32 = x_3 - x_2
    self.distvec = np.array([x_21, x_31, x_32])
  
  def x_comps(self):
    return self.distvec[:][:, 0]
  
  def y_comps(self):
    return self.distvec[:][:, 1]
  
  def estimate_x(self, delays=None):
    if delays is None:
      return 0
      
    A = 2 * np.sum(self.x_comps() ** 2)
    x = ((2 * self.c) / A) * np.sum(self.x_comps() * delays)

    return x

  def estimate_y(self, delays=None):
    if delays is None:
      return 0
      
    D = 2 * np.sum(self.y_comps() ** 2)
    y = - ((2 * self.c) / D) * np.sum(self.y_comps() * delays)
    return y
  
  def compute_angle(self, delays=None, verbose=False):
    if delays is None:
      raise ValueError('No delays provided')
    
    # x = self.estimate_x(delays)
    # y = self.estimate_y(delays)

    try:
      # theta = np.arctan2(y, x)
      theta = np.arctan2(np.sqrt(3) * (delays[0] + delays[1]), delays[0] - delays[1] - 2 * delays[2])
    except ZeroDivisionError:
      print('Illegal operation!')
      return None
    
    if verbose == True:
      print(f'Delays: {delays}')

      print(f'''Incoming angle: 
            Radians: {theta}
            Degrees: {np.degrees(theta)}
      ''')
    return theta