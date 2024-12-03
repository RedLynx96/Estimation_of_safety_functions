import numpy as np
import random
from scipy.signal import sawtooth
import tensorflow as tf
from keras.models import Model
from keras.losses import Loss
from keras import layers

# RAM CALCULATION OF THE DATASET AND SAFETY FUNCTION

class NoiseGenerator:
    """
    Generates scaled noise values from an exponential distribution with controlled maximum noise.
    """

    def __init__(self, min_value=0.001, max_value=0.1, lambda_=0.5, size=25600, seed=None):
        """
        Initializes the NoiseGenerator with specified parameters.

        Parameters:
        - min_value (float): The minimum value for scaling.
        - max_value (float): The maximum value for scaling.
        - lambda_ (float): The mean parameter for the exponential distribution.
        - size (int): The number of noise values to generate, ideally it should be same length as train_data.
        - seed (int, optional): Seed for the random number generator for reproducibility.
        """

        # Input validation
        if not (min_value < max_value):
            raise ValueError("min_value must be less than max_value.")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")
        if size <= 0 or not isinstance(size, int):
            raise ValueError("size must be a positive integer.")

        self.min_value = min_value
        self.max_value = max_value
        self.lambda_ = lambda_
        self.size = size
        self.noise_iters = 0
        
        self.scaled_values = self._generate_scaled_values()

    def _generate_scaled_values(self):
        """
        Generates and scales noise values from an exponential distribution.

        Returns:
        - np.ndarray: An array of scaled noise values.
        """
        # Generate random values from an exponential distribution with mean lambda_
        exponential_values = np.random.exponential(scale=self.lambda_, size=self.size)
        
        # Scale the values to have more values near min_value
        scaled = self.min_value + (self.max_value - self.min_value) * (1-np.exp(-exponential_values))
        
        # Ensure that scaled values do not fall below min_value
        scaled = np.maximum(scaled, self.min_value)
        
        return scaled

    def get_noise(self):
        """
        Retrieves the current maximum noise value and updates the iterator.

        Returns:
        - float: The current maximum noise value.
        """
        if self.noise_iters < self.size:
            ruido_max = self.scaled_values[self.noise_iters]
            self.noise_iters += 1
        else:
            # Regenerate scaled values and reset iterator
            self.scaled_values = self._generate_scaled_values()
            ruido_max = self.scaled_values[np.random.randint(self.size)]
            self.noise_iters = 1
        
        return ruido_max

def Function_Generator():
    """
    Generates a random function based on a combination of random base functions.
    """
    # Create Q
    Q = np.linspace(0,1,1000) # Q is defined again in Data Generator, beware 

    # Generate random coefficients
    numFuncs_A = random.randint(4, 8)  # Number of functios from set A
    numFuncs_B = random.randint(2, 4)  # Number of functios from set B
    numFuncs_C = random.randint(5, 8)  # Number of functios from set C
    numFuncs_D = random.randint(1, 2)  # Number of functios from set D

    coeffs_1 = np.random.rand(numFuncs_A + numFuncs_B + numFuncs_C)
    coeffs_2 = np.ones(numFuncs_D) * np.max(coeffs_1) / len(coeffs_1)
    coeffs = np.concatenate([coeffs_1, coeffs_2])

    # Random value for asymmetry in the tent map
    c = np.random.rand()

    # Define base functions for Ruben
    Set_A = [
        lambda x: x**2 * (1 - x)**2,
        lambda x: (np.sin(np.pi * x))**2 * x * (1 - x),
        lambda x: (4 + np.random.rand() * 2) * x * (1 - x)**2,
        lambda x: np.sqrt(x) * (1 - x)**2,
        lambda x: x**3 * (1 - x)**3,
        lambda x: (1 - np.cos(np.pi * x)) * x * (1 - x),
        lambda x: (np.tanh(4 * x - 2))**2 * x * (1 - x),
        lambda x: (1 - np.exp(-4 * x))**2 * x * (1 - x),
        lambda x: (2 + np.random.rand() * 2) * np.minimum(x / c, (1 - x) / (1 - c))**2 * x * (1 - x)
    ]

    # Define base functions 'a'
    Set_B = [
        lambda x: np.abs((1 - x**2) * (np.sin(np.pi * x))**2),
        lambda x: np.abs((x - x**2) * (np.cos(np.pi * x))**2),
        lambda x: np.abs(x * (1 - x) * (1 + x - x**2)),
        lambda x: np.abs((1 - x) * np.sin(np.pi * x) * x),
        lambda x: np.abs((x * (1 - x))**2),
        lambda x: np.abs(x**3 * (1 - x) * (2 - x)),
        lambda x: np.abs(x * (1 - x) * np.exp(-x)),
        lambda x: np.abs((1 - x) * (1 - np.cos(2 * np.pi * x))),
        lambda x: np.abs((np.sin(np.pi * x))**2 * (x**2) * (1 - x)),
        lambda x: np.abs((1 - np.exp(-x)) * (1 - x) * x),
        lambda x: np.abs(np.cos(np.pi * x) * x**2 * (1 - x)),
        lambda x: np.abs((x**2 - x) * np.cos(np.pi * x)),
        lambda x: np.abs(x * (1 - x) * (np.sin(4 * np.pi * x))**2),
        lambda x: np.abs(x * (1 - x) * (np.cos(6 * np.pi * x))**2),
        lambda x: np.abs(x * (1 - x) * ((np.sin(2 * np.pi * x))**2 + (np.cos(2 * np.pi * x))**2)),
        lambda x: np.abs(x * (1 - x) * np.exp(-10 * (x - 0.5)**2)),
        lambda x: np.abs(x * (1 - x) * (1 + np.cos(10 * np.pi * x))),
        lambda x: np.abs(x * (1 - x) * np.abs(np.sin(8 * np.pi * x))),
        lambda x: np.abs(x * (1 - x) * (1 - (np.cos(12 * np.pi * x))**2)),
        lambda x: np.abs(x * (1 - x) * (1 + (np.sin(6 * np.pi * x))**2)),
        lambda x: np.abs(x * (1 - x) * (1 + np.exp(-5 * (x - 0.3)**2) + np.exp(-5 * (x - 0.7)**2))),
        lambda x: np.abs(x * (1 - x) * (1 + np.sin(4 * np.pi * x) * np.cos(4 * np.pi * x))),
    ]

    # Define base functions 'b' (not used in selection based on MATLAB code)
    Set_C= [
        lambda x: np.abs((1 - np.exp(-5 * x)) * np.exp(-5 * (1 - x)) * x * (1 - x)),
        lambda x: np.abs(np.log(1 + 9 * x) * (1 - x)**2),
        lambda x: np.abs((x - x**2)**2 * np.sin(2 * np.pi * x)),
        lambda x: np.abs(x**2 * (1 - x) * (np.cos(np.pi * x))**2),
        lambda x: np.abs(x * (1 - x**4)),
        lambda x: np.abs(np.exp(-2 * (x - 0.5)**2) * x * (1 - x)),
        lambda x: np.abs((x * (1 - x))**3),
        lambda x: np.abs((np.sin(2 * np.pi * x) + 1) * x * (1 - x)),
        lambda x: np.abs((x * np.log(x + 1e-5)) * (1 - x)),
        lambda x: np.abs((x**0.5) * (1 - x**3)),
        lambda x: np.abs((x**3) * (1 - x**5)),
    ]

    # Define nonsmooth base functions
    Set_D = [
        lambda x: np.abs(np.sin(2 * np.pi * x) * x * (1 - x)),
        lambda x: np.abs(x * (1 - x) * ((x * 10) % 1 > 0.5)),
        lambda x: (3 + np.random.rand() * 2) * np.abs((x * 10) % 1 - 0.5) * x * (1 - x),
        lambda x: np.abs(np.floor(10 * x - 5) * (1 - x) * x),
        lambda x: (4 + np.random.rand() * 2) * np.abs(np.cos(np.pi * x)) * x * (1 - x),
        lambda x: np.abs(np.ceil(10 * x - 5) * (1 - x) * x),
        lambda x: np.abs(((x * 20) % 1 > 0.5) * (1 - x) * x),
        lambda x: np.abs(np.tan(np.pi * (x - 0.5))) * (1 - x) * x,
        lambda x: np.abs((np.floor(5 * x) % 2)) * (1 - x) * x,
        lambda x: np.abs(sawtooth(2 * np.pi * x)) * (1 - x) * x,
        lambda x: np.abs(np.sign(np.cos(2 * np.pi * x))) * (1 - x) * x,
        lambda x: np.abs((x * 50) % 1) * (1 - x) * x,
        lambda x: np.abs(np.cos(4 * np.pi * x)) * (1 - x) * x,
    ]

    # Select random base functions without replacement
    selectedFuncs_A = random.sample(Set_A, numFuncs_A)
    selectedFuncs_B = random.sample(Set_B, numFuncs_B)
    selectedFuncs_C = random.sample(Set_C, numFuncs_C)
    selectedFuncs_D = random.sample(Set_D, numFuncs_D)
    
    selectedFuncs = selectedFuncs_A + selectedFuncs_B + selectedFuncs_C + selectedFuncs_D

    # Generate y
    y = np.zeros_like(Q)
    for i, func in enumerate(selectedFuncs):
        y += coeffs[i] * func(Q)

    # Scale y so that the maximum value is between 1 and 1.5
    maxVal = np.max(y)
    scaleFactor = 1 + np.random.rand() * 0.5  # Scale factor between 1 and 1.5
    y = y / maxVal * scaleFactor

    # Ensure that at least one value in y exceeds 1.2
    while np.max(y) <= 1.2:
        y = y * (1 + np.random.rand() * 0.2)  # Scale by a factor between 1 and 1.2

    return y

def Data_Generator(y, ruido_max, n_series, padding, mean_extension, test, b, separated_orbits=True):
    """
    Generates data based on a combination of random base functions with added noise.

    Parameters:
    ----------
    y: numpy.ndarray
        The map function to generate the data.
    ruido_max : float
        The maximum noise value to be added to the generated signal.
    n_series : int
        The number orbits to sample from the system
    padding : int
        The maximum number of points in each orbit.
    mean_extension : int
        Length of the time series to generate, by either padding -1 values, or truncating the series.
    test : bool
        Whether to generate data for testing or training.
    b : int
        The index of the safety function to use for testing.
    separated_orbits : bool
        Whether to separate each orbit with a [-1, -1] separator.

    Returns:
    -------
    xVectors : numpy.ndarray
        The sampled values from the orbit as an 2D array (n_series, padding).
    Us : numpy.ndarray
        The control matrix computed through the safety functions algorithm.
    """
    
    Q = np.linspace(0, 1, 1000) # Q is defined again in Data Generator, beware
    iteraciones = 50 # Number of iterations for the safety functions algorithm
    escala = 0.001
    
    # Generate noise
    ruido = np.arange(-ruido_max, ruido_max + escala, escala)
    Nruido = len(ruido)
    anchoruido = Nruido // 2  # Anchor point for noise

    Imagen_ruido = np.zeros((len(Q), Nruido)) # 1000 is the number of pointsi in Q

    for i in range(len(Q)):
        for s in range(Nruido):
            Imagen_ruido[i, s] = y[i] + ruido[s]

    ymin = np.min(Imagen_ruido) - escala
    ymax = np.max(Imagen_ruido) + escala

    Imagen_ruidoy = np.arange(ymin, ymax + escala, escala)
    Nimagen_ruidoy = len(Imagen_ruidoy)
    controly = np.zeros(Nimagen_ruidoy)

    # Safety Functions Algorithm
    Us = np.zeros((iteraciones, len(Q)))

    IndImagen = np.round((y - ymin) / escala).astype(int)

    for k in range(iteraciones-1):
        for m in range(Nimagen_ruidoy):
            controly_posibles_inQ = np.abs(Imagen_ruidoy[m] - Q)
            controly[m] = np.min(np.maximum(Us[k, :], controly_posibles_inQ))

        for i in range(len(Q)):
            Us[k + 1, i] = np.max(controly[IndImagen[i] - anchoruido:IndImagen[i] + anchoruido])

        if np.array_equal(Us[k + 1, :], Us[k, :]):
            Us[k + 1:, :] = Us[k + 1, :]
            #print('Converge en la iteración: ', k)
            break

    Us = Us[-1,:] 

    # Initialize a matrix to store the time series with noise
    xVectors = -1 * np.ones((n_series, padding))  # Initialize with -1 values
    xVectors[:, 0] = np.random.rand(n_series)

    for j in range(n_series):
        noise = np.random.uniform(-ruido_max, ruido_max, len(Q))
        for i in range(padding - 1):
            xVectors[j, i + 1] = np.interp(xVectors[j, i], Q, y) + noise[i]
            if xVectors[j, i + 1] > 1 or xVectors[j, i + 1] < 0:  # Modify the condition to break when above 1 or below 0
                break
    
    if test == False:
        Ic = random.randint(0, Us.shape[0]-1)
    else:
        Ic = b

    Us = Us[Ic] 

    xVectors= np.reshape(xVectors, (int(n_series* padding), 1))
    shifted_column = np.roll(xVectors, shift=-1, axis=0)
    shifted_column[-1, 0] = -1

    xVectors = np.hstack((xVectors, shifted_column))   
    xVectors = xVectors[~np.any(xVectors == -1, axis=1)]
 
    if separated_orbits == True:
        # We separate each orbit with a [-1, -1] separator
        Separated_xVectors = []
        for i in range(len(xVectors) - 1):
            Separated_xVectors.append(xVectors[i])
            if xVectors[i][1] != xVectors[i + 1][0]:
                Separated_xVectors.append([-1, -1])

        # Append the last row
        Separated_xVectors.append(xVectors[-1])

        # Convert result back to numpy array
        Separated_xVectors = np.array(Separated_xVectors)

        xVectors = Separated_xVectors

    if xVectors.shape[0] > mean_extension:
        xVectors = xVectors[:mean_extension, :]
    elif xVectors.shape[0] < mean_extension:
        num_padding_rows = mean_extension - xVectors.shape[0]
        padding = np.full((num_padding_rows, 2), -1)
        xVectors = np.vstack((xVectors, padding))
    
    return xVectors, Us, Ic

# Define the learning rate schedule function
def lr_warmup_scheduler(epoch, lr, initial_lr ,warmup_epochs, total_epochs, decay_factor):
    if epoch < warmup_epochs:
        # Linear warmup: increase learning rate
        learning_rate = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Exponential decay: reduce learning rate after warmup
        learning_rate = initial_lr * decay_factor ** (((epoch + 1) - warmup_epochs) / (total_epochs - warmup_epochs))

    return learning_rate

# Define the weighted MSE loss function
class AsymmetricMSELoss(Loss):
    def __init__(self, underestimation_weight=2.0, reduction=tf.keras.losses.Reduction.AUTO, name="asymmetric_mse_loss"):
        super().__init__(reduction=reduction, name=name)
        self.underestimation_weight = tf.constant(underestimation_weight, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # Calculate the error
        error = y_true - y_pred
        
        # Penalize underestimation more by applying a weight
        weight = tf.where(error > 0, self.underestimation_weight, tf.constant(1.0, dtype=tf.float32))
        
        # Calculate the MSE with the asymmetric weighting
        mse = tf.reduce_mean(weight * tf.square(error))
        
        # Return MSE^2
        return mse
    
# FUNCTIONS TO CALCULATE THE SAFETY FUNCTIONS FROM A PREVIOUSLY GENERATED DATASET OF .MAT FILES
def Extended_Function_Dataset(data, mean_extension, test, b, separated_orbits):
    Us = data['Us'][-1,:]
    xVectors = data['xVectors']

    if test == False:
        Ic = random.randint(0, Us.shape[0]-1)
    else:
        Ic = b

    Us = Us[Ic] #quitarlo para hacer cheating

    xVectors = np.nan_to_num(xVectors, nan=-1) # hay algunos nan de matlab
    xVectors= np.reshape(xVectors, (1000* 50, 2))
    xVectors = xVectors[~np.any(xVectors == -1, axis=1)]

    if separated_orbits == True:
        # We separate each orbit with a [-1, -1] separator
        Separated_xVectors = []
        for i in range(len(xVectors) - 1):
            Separated_xVectors.append(xVectors[i])
            if xVectors[i][1] != xVectors[i + 1][0]:
                Separated_xVectors.append([-1, -1])
        # Append the last row
        Separated_xVectors.append(xVectors[-1])
        # Convert result back to numpy array
        Separated_xVectors = np.array(Separated_xVectors)
        xVectors = Separated_xVectors

    if xVectors.shape[0] > mean_extension:
        xVectors = xVectors[:mean_extension, :]
    elif xVectors.shape[0] < mean_extension:
        num_padding_rows = mean_extension - xVectors.shape[0]
        padding = np.full((num_padding_rows, 2), -1)
        xVectors = np.vstack((xVectors, padding))
    
    return xVectors, Us, Ic