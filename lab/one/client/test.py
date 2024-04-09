import numpy as np

tolerance = 1

goal = 100_000 / 910_000

i = np.arange(0, 1, 0.01)

def f(x):
  for j in range(21):
    temp = 1 / ((1 + x) ** j)

result = f(i)

print('done:', result)