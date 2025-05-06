import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate sample data for demonstration purposes"""
    dates = pd.date_range(start='2024-01-01', end='2024-04-30', freq='H')
    data = {
        'timestamp': dates,
        'FCR_price': np.sin(np.arange(len(dates)) / 24 * np.pi) * 10 + 50 + np.random.normal(0, 5, len(dates)),
        'aFRR_up_price': np.sin(np.arange(len(dates)) / 24 * np.pi + 1) * 15 + 70 + np.random.normal(0, 7, len(dates)),
        'aFRR_down_price': np.sin(np.arange(len(dates)) / 24 * np.pi - 1) * 12 + 30 + np.random.normal(0, 6, len(dates)),
        'grid_frequency': 50 + np.sin(np.arange(len(dates)) / 4) * 0.05 + np.random.normal(0, 0.01, len(dates)),
        'renewable_generation': np.sin(np.arange(len(dates)) / 24 * np.pi) * 5000 + 10000 + np.random.normal(0, 1000, len(dates)),
        'load': np.sin(np.arange(len(dates)) / 24 * np.pi) * 8000 + 30000 + np.random.normal(0, 2000, len(dates)),
        'battery_soc': np.clip(50 + np.cumsum(np.random.normal(0, 5, len(dates))) % 80, 10, 90)
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the data for analysis"""
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in processed_df.columns and not pd.api.types.is_datetime64_any_dtype(processed_df['timestamp']):
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
    
    # Add time-based features
    processed_df['hour'] = processed_df['timestamp'].dt.hour
    processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
    processed_df['month'] = processed_df['timestamp'].dt.month
    processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Handle missing values if any
    numeric_cols = processed_df.select_dtypes(include=['number']).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
    
    return processed_df

def create_feature_matrix(df, target_col, lookback_hours=24):
    """Create feature matrix for time series prediction with lookback window"""
    # Create lagged features
    features_df = df.copy()
    
    # Add lag features
    for i in range(1, lookback_hours + 1):
        features_df[f'{target_col}_lag_{i}'] = features_df[target_col].shift(i)
    
    # Add rolling statistics
    for window in [6, 12, 24]:
        features_df[f'{target_col}_rolling_mean_{window}'] = features_df[target_col].rolling(window=window).mean().shift(1)
        features_df[f'{target_col}_rolling_std_{window}'] = features_df[target_col].rolling(window=window).std().shift(1)
    
    # Drop rows with NaN values (due to lag/rolling features)
    features_df = features_df.dropna()
    
    return features_df