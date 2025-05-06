import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def predict_prices(df, features, target, prediction_hours=24):
    """
    Simple prediction model for price prediction
    
    Args:
        df: DataFrame with historical data
        features: List of feature column names
        target: Target column name to predict
        prediction_hours: Number of hours to predict ahead
        
    Returns:
        Array of predictions
    """
    # Prepare data
    X = df[features].values
    y = df[target].values
    
    # Train on all data except last prediction_hours
    X_train = X[:-prediction_hours]
    y_train = y[:-prediction_hours]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predict next prediction_hours
    X_pred = X[-prediction_hours:]
    X_pred_scaled = scaler.transform(X_pred)
    predictions = model.predict(X_pred_scaled)
    
    return predictions

def calculate_confidence_interval(predictions, actual_values, confidence_level=95):
    """
    Calculate confidence intervals for predictions
    
    Args:
        predictions: Array of predicted values
        actual_values: Array of actual values (for calculating error distribution)
        confidence_level: Confidence level percentage (e.g., 95 for 95% CI)
        
    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays
    """
    # Calculate prediction errors
    errors = actual_values - predictions
    
    # Calculate standard deviation of errors
    error_std = np.std(errors)
    
    # Z-score for the given confidence level
    z_scores = {
        90: 1.645,
        95: 1.96,
        99: 2.576
    }
    z_score = z_scores.get(confidence_level, 1.96)
    
    # Calculate confidence intervals
    lower_bounds = predictions - z_score * error_std
    upper_bounds = predictions + z_score * error_std
    
    return lower_bounds, upper_bounds

def evaluate_predictions(actual, predicted):
    """
    Calculate evaluation metrics for predictions
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    # Avoid division by zero
    actual_nonzero = np.where(actual == 0, 1e-10, actual)
    mape = np.mean(np.abs((actual - predicted) / actual_nonzero)) * 100
    
    max_error = np.max(np.abs(actual - predicted))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MaxError': max_error
    }

def calculate_battery_revenue(prices, battery_power, battery_capacity):
    """
    Calculate potential revenue from battery operations
    
    Args:
        prices: Array of hourly prices
        battery_power: Battery power in MW
        battery_capacity: Battery capacity in MWh
        
    Returns:
        Dictionary with revenue metrics and operation schedule
    """
    # Sort prices to find best hours to charge/discharge
    sorted_prices = pd.Series(prices).sort_values()
    battery_hours = min(int(battery_capacity / battery_power), len(prices) // 2)
    
    # Determine charge and discharge hours
    charge_hours = sorted_prices.iloc[:battery_hours].index
    discharge_hours = sorted_prices.iloc[-battery_hours:].index
    
    # Create operation schedule
    schedule = pd.DataFrame({
        'hour': range(len(prices)),
        'price': prices,
        'operation': 'idle'
    })
    
    for hour in charge_hours:
        schedule.loc[schedule['hour'] == hour, 'operation'] = 'charge'
    
    for hour in discharge_hours:
        schedule.loc[schedule['hour'] == hour, 'operation'] = 'discharge'
    
    # Calculate revenue
    buy_price = schedule.loc[schedule['operation'] == 'charge', 'price'].sum() * battery_power
    sell_price = schedule.loc[schedule['operation'] == 'discharge', 'price'].sum() * battery_power
    revenue = sell_price - buy_price
    
    # Calculate state of charge
    soc = np.zeros(len(prices) + 1)  # +1 for initial SOC
    soc[0] = 50  # Start at 50% SOC
    
    for i, row in schedule.iterrows():
        if row['operation'] == 'charge':
            # Charge at full power if possible
            energy_change = min(battery_power, battery_capacity - soc[i])
            soc[i+1] = soc[i] + energy_change
        elif row['operation'] == 'discharge':
            # Discharge at full power if possible
            energy_change = min(battery_power, soc[i])
            soc[i+1] = soc[i] - energy_change
        else:
            soc[i+1] = soc[i]
    
    # Add SOC to schedule
    schedule['soc'] = soc[1:]
    
    return {
        'revenue': revenue,
        'daily_revenue': revenue * (24 / len(prices)),
        'monthly_revenue': revenue * (24 * 30 / len(prices)),
        'annual_revenue': revenue * (24 * 365 / len(prices)),
        'schedule': schedule
    }