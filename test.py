
import numpy as np

def normalize(val, min_val, max_val):

    return (val - min_val) / (max_val - min_val)


def linear_up(epoch):

    normal = normalize(epoch, 0.0, 29.0)

    return normal * 5.0

def linear_down(epoch):
    
    normal = normalize(epoch, 0.0, 29.0)

    print normal

    return 5.0 - (normal * 5.0)

def quadratic(epoch):

    normal = normalize(epoch, 0.0, 29.0)

    return 5.0 * ((-4.0 * normal**2) + 4.0 * normal)

def sigmoid_inverse(epoch):

    """The sigmoid function."""

    normal = normalize(epoch, 0.0, 29.0)

    print(normal)

    return 5 * (1.0/(1.0+np.exp(13 * (normal - 0.5))))

def sigmoid_scaled(epoch):       

    """The sigmoid function."""

    normal = normalize(epoch, 0.0, 29.0)

    return 5 * (1.0/(1.0+np.exp(-13 * (normal - 0.5))))


    
if __name__ == "__main__":

    print linear_down(10)
    print linear_up(10)
    print quadratic(15)
    print sigmoid_inverse(10)
    print sigmoid_scaled(10)
