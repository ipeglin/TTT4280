import numpy as np

def sample_mean_estimate(sample):
  return sum(sample) / len(sample)

def sample_variance_estimate(sample):
  mean_est = sample_mean_estimate(sample)
  return np.sqrt((1 / (len(sample) - 1)) * sum([(x - mean_est)**2 for x in sample]))