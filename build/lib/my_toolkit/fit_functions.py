import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Polynomials
def linear(x, slope, intercept):
    return slope * x + intercept

def square_root(x, scale, offset):
    return scale * np.sqrt(x) + offset

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def quartic(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def quintic(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f


# Non-polynomial functions
def logistic(x, a, b, c, d):
    return a / (1 + np.exp(-c * (x - d))) + b

def logarithmic(x, a, b):
    return a * np.log(x) + b

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

def hyperbolic(x, a, b, c):
    return a / (x + b) + c

def power_law(x, a, b):
    return a * x**b

# Periodic functions
def sine(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def cosine(x, a, b, c, d):  
    return a * np.cos(b * x + c) + d

def model_dict():
    return {"linear": {"func": linear, "expression": "m * x + b"},
            "square_root": {"func": square_root, "expression": "a * sqrt(x) + b"},
            "quadratic": {"func": quadratic, "expression": "a * x^2 + b * x + c"},
            "cubic": {"func": cubic, "expression": "a * x^3 + b * x^2 + c * x + d"},
            "quartic": {"func": quartic, "expression": "a * x^4 + b * x^3 + c * x^2 + d * x + e"},
            "quintic": {"func": quintic, "expression": "a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f"},
            "logistic": {"func": logistic, "expression": "a / (1 + exp(-c * (x - d))) + b"},
            "logarithmic": {"func": logarithmic, "expression": "a * log(x) + b"},
            "exponential": {"func": exponential, "expression": "a * exp(b * x) + c"},
            "gaussian": {"func": gaussian, "expression": "a * exp(-((x - b)^2) / (2 * c^2))"},
            "hyperbolic": {"func": hyperbolic, "expression": "a / (x + b) + c"},
            "power_law": {"func": power_law, "expression": "a * x^b"},
            "sine": {"func": sine, "expression": "a * sin(b * x + c) + d"},
            "cosine": {"func": cosine, "expression": "a * cos(b * x + c) + d"},
            }

def estimate_frequency_peaks(x_data, y_data, distance=None, prominence=None):
    """
    Estimate frequency using peaks in the data with intelligent tuning.

    Args:
        x_data (np.ndarray): Independent variable data.
        y_data (np.ndarray): Dependent variable data.
        distance (int, optional): Minimum number of samples between peaks.
        prominence (float, optional): Minimum prominence of peaks.

    Returns:
        float: Estimated frequency.

    Raises:
        ValueError: If not enough peaks are found to estimate frequency.
    """
    # Automatically set the `distance` if not provided
    if distance is None:
        distance = len(x_data) // 10  # Adjust based on expected periodicity

    # Automatically set the `prominence` if not provided
    if prominence is None:
        prominence = (np.max(y_data) - np.min(y_data)) / 10  # 10% of the range

    # Detect peaks with the given parameters
    peaks, properties = find_peaks(y_data, distance=distance, prominence=prominence)

    # Check if enough peaks are detected
    if len(peaks) < 2:
        raise ValueError("Not enough peaks to estimate frequency. Adjust parameters.")

    # Calculate distances between consecutive peaks in x_data
    peak_distances = np.diff(x_data[peaks])
    avg_distance = np.mean(peak_distances)

    # Frequency is the reciprocal of the average distance between peaks
    frequency = 1 / avg_distance
    return frequency

def estimate_params_fft(t, y):
    # Estimate amplitude (A) as half the difference between max and min
    A = (np.max(y) - np.min(y)) / 2
    
    # Estimate vertical offset (D) as the average of max and min values
    D = (np.max(y) + np.min(y)) / 2
    
    # Perform the Fourier Transform
    n = len(y)
    fft_y = np.fft.fft(y - D)  # Subtract D to focus on the oscillation part of the signal
    fft_freq = np.fft.fftfreq(n, t[1] - t[0])  # Frequency bins corresponding to FFT
    
    # Remove the negative frequencies (we only care about positive frequencies)
    fft_y = fft_y[:n//2]
    fft_freq = fft_freq[:n//2]
    
    # Find the peak frequency in the FFT spectrum
    peak_idx = np.argmax(np.abs(fft_y))
    B = 2 * np.pi * fft_freq[peak_idx]
    
    # Estimate phase shift (C) by checking the phase of the dominant frequency
    phase = np.angle(fft_y[peak_idx])
    
    # Estimate the phase shift using the angle of the complex number
    C = phase if np.isnan(phase) == False else 0  # Default to zero if phase is NaN
    
    return A, B, C, D

def guess_parameters(x_data, y_data, model):
    """
    Intelligently guesses initial parameters for various mathematical models.

    Args:
        x_data (np.ndarray): Independent variable data.
        y_data (np.ndarray): Dependent variable data.
        model (str): The name of the model. Options are:
            "linear", "square_root", "quadratic", "cubic", "quartic",
            "logistic", "logarithmic", "exponential", "sine", "cosine".

    Returns:
        list: A list of initial parameter guesses for the specified model.
    """
    # General statistics
    y_min, y_max = np.min(y_data), np.max(y_data)
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_range = y_max - y_min
    x_range = x_max - x_min
    mean_x, mean_y = np.mean(x_data), np.mean(y_data)
    
    if model == "linear":
        slope = (y_max - y_min) / (x_max - x_min)
        intercept = y_min - slope * x_min
        return [slope, intercept]
    
    elif model == "square_root":
        scale = y_range / np.sqrt(x_range)
        offset = y_min
        return [scale, offset]
    
    elif model == "quadratic":
        coeff_a = (y_data[-1] - y_data[0]) / (x_data[-1]**2 - x_data[0]**2)
        coeff_b = 0  # Assuming symmetric curve
        coeff_c = y_min
        return [coeff_a, coeff_b, coeff_c]
    
    elif model == "cubic":
        coeff_a = (y_data[-1] - y_data[0]) / (x_data[-1]**3 - x_data[0]**3)
        coeff_b = 0  # Assuming symmetric cubic curve
        coeff_c = 0
        coeff_d = y_min
        return [coeff_a, coeff_b, coeff_c, coeff_d]
    
    elif model == "quartic":
        coeff_a = (y_data[-1] - y_data[0]) / (x_data[-1]**4 - x_data[0]**4)
        coeff_b, coeff_c, coeff_d = 0, 0, 0
        coeff_e = y_min
        return [coeff_a, coeff_b, coeff_c, coeff_d, coeff_e]
    
    elif model == "quintic":
        coeff_a = (y_data[-1] - y_data[0]) / (x_data[-1]**4 - x_data[0]**4)
        coeff_b, coeff_c, coeff_d, coeff_e = 0, 0, 0, 0
        coeff_f = y_min
        return [coeff_a, coeff_b, coeff_c, coeff_d, coeff_e, coeff_f]
    
    elif model == "logistic":
        midpoint = x_data[len(x_data) // 2]
        growth_rate = 1.0 / x_range
        scale = y_range
        baseline = y_min
        return [midpoint, growth_rate, scale, baseline]
    
    elif model == "logarithmic":
        scale = y_range / np.log(x_max - x_min + 1)
        offset = y_min
        return [scale, offset]
    
    elif model == "exponential":
        rate = np.log(y_data[-1] / y_data[0]) / x_range
        scale = y_data[0] / np.exp(rate * x_min)
        offset = y_min
        return [scale, rate, offset]
    
    elif model == "gaussian":
        a = y_range
        b = mean_x
        c = x_range / 4
        return [a, b, c]
    
    elif model == "hyperbolic":
        a = y_range
        b = mean_x
        c = y_min
        return [a, b, c]
    
    elif model == "power_law":
        a = y_range / (x_range ** 2)
        b = 2
        return [a, b]
    
    elif model == "sine" or model == "cosine":
        amplitude, frequency, phase, offset = estimate_params_fft(x_data, y_data)
       
        return [amplitude, frequency, phase, offset]
    
    else:
        raise ValueError(f"Unsupported model: {model}")


def parse_results(fit_result, verbose = True):
    """
    Parses the fitting results into a human-readable format.

    Args:
        fit_result (dict): Fitting results dictionary.

    Returns:
        str: A formatted string summarizing the fitting results.
    """
    modeldict = model_dict()
    best_model_name = fit_result["best_model"]
    best_model_expression = modeldict[best_model_name]["expression"]
    best_params = fit_result["fit_result"]["parameters"]
    r_squared = fit_result["fit_result"]["r_squared"]

    if verbose:
        print(f"Best model: {best_model_name} \t{best_model_expression} \nParameters: {best_params}\nR-squared: {r_squared}")
        
    return best_model_name, best_params

# Function to fit a dataset to a specified model
def fit_function(x_data, y_data, model=linear, p0 = None):
    """
    Fits the provided data to the specified model.

    Args:
        x_data (list or np.ndarray): Independent variable data.
        y_data (list or np.ndarray): Dependent variable data.
        model (callable): A model function to fit (e.g., linear, quadratic).

    Returns:
        dict: A dictionary with fitted parameters, their covariance, and R-squared value.
    """
    # Ensure input is in numpy array format
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Perform curve fitting
    params, covariance = curve_fit(model, x_data, y_data, p0 = p0)
    
    # Predicted y values from the model
    y_pred = model(x_data, *params)

    # Calculate R-squared
    residuals = y_data - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        "parameters": params,
        "covariance": covariance,
        "r_squared": r_squared
    }

def plot_fit(x_data, y_data, fit_func, popt):
    """
    Plot the fit results.

    Args:
        x_data (np.ndarray): Independent variable data.
        y_data (np.ndarray): Dependent variable data.
        fit_func (callable): Fitting function.
        popt (list): Optimal parameters for the fitting function.
    """
    plt.figure(figsize = (10,6))
    plt.title("Automatic function fitting")
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, fit_func(x_data, *popt), label='Fit', color='red')
    plt.legend()
    plt.show()

# Example of fitting multiple models
def find_best_fit(x_data, y_data, plot = False):
    """
    Tries multiple models and selects the best-fitting one.

    Args:
        x_data (list or np.ndarray): Independent variable data.
        y_data (list or np.ndarray): Dependent variable data.

    Returns:
        dict: Best-fit model and its fitting results.
    """
    models = model_dict()
    results = {}

    for name, model in models.items():
        try:
            p0 = guess_parameters(x_data, y_data, name)
                
            fit_result = fit_function(x_data, y_data, model=model["func"], p0 = p0)
            results[name] = fit_result
        except RuntimeError:
            # Handle fitting errors (e.g., insufficient data or bad model)
            results[name] = None

    # Select the model with the highest R-squared
    best_model = max(
        results.items(), key=lambda item: item[1]["r_squared"] if item[1] else -np.inf
    )

    if plot:
        plot_fit(x_data, y_data, models[best_model[0]]["func"], best_model[1]["parameters"])
        
    return {"best_model": best_model[0], "fit_result": best_model[1]}

