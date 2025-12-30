# xavier is used for initialization of sigmoid and tanh initialization functions
import numpy as np
def xavier_init(fan_in,fan_out):
  return np.random.randn(fan_in,fan_out) * np.sqrt(1.0 / fan_in)

