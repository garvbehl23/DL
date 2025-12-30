# he is used for initialization of relu activation function

def he_init(fan_in, fan_out):
    return np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)

# we used it due to the LeakyRelu problem
