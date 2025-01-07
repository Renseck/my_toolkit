import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

def get_method_kwargs(method):
    method_kwargs = {
        "rolling z-score": ["window (int)", "threshold (float)", "return_all (bool)"],
        "rolling iqr": ["window (int)", "replace (bool)"],
        "hampel": ["window (int)", "n_sigmas (int)", "replace (bool)"],
        "dbscan": ["eps (float)", "min_samples (int)"],
        "seasonal decomposition": ["seasonal (int)", "trend (int)", "threshold (int)", "replace (bool)"]
    }
    return method_kwargs.get(method, [])

def remove_outliers(dataframe: pd.DataFrame, method: str = "quantile", help = False, **kwargs) -> pd.DataFrame:
    """
    Remove outliers from a pandas dataframe.

    Parameters:
    dataframe (pd.DataFrame): The dataframe from which to remove outliers.
    method (str): The method to use for outlier removal. Default is "quantile".

    Returns:
    pd.DataFrame: The dataframe with outliers removed.
    """
    available_methods = ["quantile", "z-score", "rolling z-score", "iqr", "rolling iqr", "hampel", "dbscan", "seasonal decomposition"]
    
    if help:
        print(f"Available kwargs for method '{method}': {get_method_kwargs(method)}")
        return
    
    if method == "quantile":
        return dataframe[(dataframe < dataframe.quantile(0.95)) & (dataframe > dataframe.quantile(0.05))]
    
    elif method == "z-score":
        treshold = kwargs.get("threshold", 3)
        return dataframe[(dataframe - dataframe.mean()).abs() < treshold * dataframe.std()]
    
    elif method == "rolling z-score":
        window = kwargs.get("window", 5)
        threshold = kwargs.get("thresh", 2)
        return_all = kwargs.get("return_all", False)
        
        rolling_window = dataframe.rolling(window = window, min_periods = 1, center = True)
        average = rolling_window.mean()
        std = rolling_window.std()
        z = dataframe.sub(average).div(std)
        m = z.between(-threshold, threshold)
        
        if return_all:
            return z, average, std, m
        return dataframe.where(m, average)
    
    elif method == "iqr":
        q1 = dataframe.quantile(0.25)
        q3 = dataframe.quantile(0.75)
        iqr = q3 - q1
        return dataframe[(dataframe > q1 - 1.5 * iqr) & (dataframe < q3 + 1.5 * iqr)]
    
    elif method == "rolling iqr":
        window = kwargs.get("window", 5)
        replace = kwargs.get("replace", False)
        
        rolling_q1 = dataframe.rolling(window = window).quantile(0.25)
        rolling_q3 = dataframe.rolling(window = window).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        
        # Tie these to a user input?
        lower_bound = rolling_q1 - 1.5 * rolling_iqr
        upper_bound = rolling_q3 + 1.5 * rolling_iqr
        
        outliers = (dataframe < lower_bound) | (dataframe > upper_bound)
        
        # Replace outliers with rolling median
        if replace:
            dataframe[outliers] = dataframe.rolling(window = window, center = True).median()[outliers]
            return dataframe
        
        return dataframe[~outliers]
    
    elif method == "hampel":
        window = kwargs.get("window" , 5)
        n_sigmas = kwargs.get("n_sigmas", 3)
        replace = kwargs.get("replace", False)
        
        rolling_median = dataframe.rolling(window = window, center = True).median()
        rolling_mad = dataframe.rolling(window=window, center=True).apply(lambda x: (x - x.median()).abs().median())
        
        # Identifity outliers
        threshold = n_sigmas * rolling_mad
        outliers = (dataframe - rolling_median).abs() > threshold
        
        if replace:
            dataframe[outliers] = rolling_median[outliers]
            
        return dataframe[~outliers]
    
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        
        ts_reshaped = dataframe.values.reshape(-1, 1)
        
        # Apply DBScan
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        labels = dbscan.fit_predict(ts_reshaped)
        
        # Remove outliers and return
        return dataframe[labels != -1]
        
    elif method == "seasonal decomposition":
        seasonal = kwargs.get("seasonal", 7)
        trend = kwargs.get("trend", None)
        threshold = kwargs.get("threshold", 3)
        period = kwargs.get("period", None)
        replace = kwargs.get("replace", False)
        
        decomposition = STL(dataframe, seasonal = seasonal, trend = trend, period = period).fit()
        residuals = decomposition.resid
        
        z_scores = (residuals - residuals.mean()) / residuals.std()
        
        outliers = np.abs(z_scores) > threshold
        
        if replace:
            dataframe[outliers] = dataframe.median()
            return dataframe
            
        return dataframe[~outliers]
    
    elif method not in available_methods:
        raise ValueError(f"Method {method} not recognized. Please choose from {available_methods}.")
    
    

def show_outliers(original_dataframe, cleaned_dataframe):
    """
    Show the outliers that were removed from a dataframe.

    Parameters:
    original_dataframe (pd.DataFrame): The original dataframe.
    cleaned_dataframe (pd.DataFrame): The cleaned dataframe.
    """
    outliers = original_dataframe[~original_dataframe.index.isin(cleaned_dataframe.index)]
    
    plt.figure(figsize = (10, 6))
    plt.title("Showing cleaned data")
    plt.scatter(original_dataframe.index, original_dataframe, label = "Original data")
    plt.scatter(outliers.index, outliers, label = "Outliers", color = "red")
    plt.legend()
    plt.show()