import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Default retro gaming colors
COLORS = {
    'background': 'black',
    'text': '#FFD700',  # Golden yellow
    'primary': '#FF7F50',  # Coral
    'secondary': '#00FF00',  # Green
    'tertiary': '#00CCFF',  # Cyan
    'grid': '#333333',
}

def create_price_trend_chart(df, price_columns, title="Price Trends"):
    """
    Create a price trend line chart with retro gaming style
    
    Args:
        df: DataFrame with timestamp and price columns
        price_columns: List of price column names to plot
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    color_map = {
        price_columns[0]: COLORS['text'],
        price_columns[1] if len(price_columns) > 1 else '': COLORS['primary'],
        price_columns[2] if len(price_columns) > 2 else '': COLORS['secondary'],
    }
    
    for col in price_columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[col],
            mode='lines',
            name=col,
            line=dict(color=color_map.get(col, COLORS['text']), width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (€/MWh)',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(
            family="VT323",
            size=16,
            color=COLORS['text']
        ),
        legend=dict(
            font=dict(
                family="VT323",
                size=14,
                color=COLORS['text']
            ),
            bgcolor='rgba(0, 0, 0, 0.5)',
            bordercolor=COLORS['primary']
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=COLORS['primary'],
            linewidth=2,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=COLORS['primary'],
            linewidth=2,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

def create_price_heatmap(df, value_column, title="Price Heatmap"):
    """
    Create a price heatmap by day and hour with retro gaming style
    
    Args:
        df: DataFrame with timestamp and price column
        value_column: Column name for values to display in heatmap
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    # Prepare data for heatmap
    heatmap_data = df.copy()
    heatmap_data['day'] = heatmap_data['timestamp'].dt.day_name()
    heatmap_data['hour'] = heatmap_data['timestamp'].dt.hour
    
    # Calculate average prices by day and hour
    pivot_data = heatmap_data.pivot_table(
        index='day', 
        columns='hour',
        values=value_column,
        aggfunc='mean'
    )
    
    # Order days correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        color_continuous_scale=[COLORS['background'], COLORS['text'], COLORS['primary']],
        labels=dict(x="Hour of Day", y="Day of Week", color=value_column),
        x=list(range(24)),
        y=day_order
    )
    
    fig.update_layout(
        title=title,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(
            family="VT323",
            size=16,
            color=COLORS['text']
        ),
        coloraxis_colorbar=dict(
            title=value_column,
            thicknessmode="pixels",
            thickness=20,
            tickfont=dict(
                family="VT323",
                size=14,
                color=COLORS['text']
            )
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

def create_prediction_chart(actual, predicted, timestamps, confidence_bounds=None, title="Price Prediction"):
    """
    Create a prediction chart with actual vs predicted values and optional confidence interval
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        timestamps: Array of timestamps
        confidence_bounds: Tuple of (lower_bounds, upper_bounds) arrays for confidence interval
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color=COLORS['text'], width=3),
        marker=dict(size=8, color=COLORS['text'])
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8, color=COLORS['primary'])
    ))
    
    # Confidence interval if provided
    if confidence_bounds is not None:
        lower_bounds, upper_bounds = confidence_bounds
        
        fig.add_trace(go.Scatter(
            x=timestamps.tolist() + timestamps.tolist()[::-1],
            y=upper_bounds.tolist() + lower_bounds.tolist()[::-1],
            fill='toself',
            fillcolor=f'rgba({int(COLORS["primary"][1:3], 16)}, {int(COLORS["primary"][3:5], 16)}, {int(COLORS["primary"][5:7], 16)}, 0.2)',
            line=dict(color='rgba(255, 127, 80, 0)'),
            name='Confidence Interval'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price (€/MWh)',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(
            family="VT323",
            size=16,
            color=COLORS['text']
        ),
        legend=dict(
            font=dict(
                family="VT323",
                size=14,
                color=COLORS['text']
            ),
            bgcolor='rgba(0, 0, 0, 0.5)',
            bordercolor=COLORS['primary']
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=COLORS['primary'],
            linewidth=2,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=COLORS['primary'],
            linewidth=2,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

def create_battery_operation_chart(schedule, battery_power, battery_capacity, title="Battery Operation Strategy"):
    """
    Create a chart showing battery operation strategy
    
    Args:
        schedule: DataFrame with hour, price, operation, and soc columns
        battery_power: Battery power in MW
        battery_capacity: Battery capacity in MWh
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=schedule['hour'],
        y=schedule['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color=COLORS['text'], width=2),
        marker=dict(size=8),
        yaxis='y'
    ))
    
    # SOC line
    fig.add_trace(go.Scatter(
        x=schedule['hour'],
        y=schedule['soc'],
        mode='lines+markers',
        name='Battery SOC (MWh)',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Operation bars
    operation_colors = {
        'charge': COLORS['secondary'],  # Green
        'discharge': '#FF0000',  # Red
        'idle': '#808080'  # Gray
    }
    
    for operation in operation_colors.keys():
        mask = schedule['operation'] == operation
        if mask.any():
            fig.add_trace(go.Bar(
                x=schedule.loc[mask, 'hour'],
                y=[battery_power if operation == 'discharge' else -battery_power if operation == 'charge' else 0] * mask.sum(),
                name=operation.capitalize(),
                marker_color=operation_colors[operation],
                yaxis='y3'
            ))
    
    fig.update_layout(
        title=title,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(
            family="VT323",
            size=16,
            color=COLORS['text']
        ),
        legend=dict(
            font=dict(
                family="VT323",
                size=14,
                color=COLORS['text']
            ),
            bgcolor='rgba(0, 0, 0, 0.5)',
            bordercolor=COLORS['primary']
        ),
        xaxis=dict(
            title='Hour',
            showgrid=True,
            gridcolor=COLORS['grid'],
            tickmode='linear',
            tick0=0,
            dtick=1,
            gridwidth=1,
            showline=True,
            linecolor=COLORS['primary'],
            linewidth=2,
        ),
        yaxis=dict(
            title='Price (€/MWh)',
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=COLORS['primary'],
            linewidth=2,
        ),
        yaxis2=dict(
            title='Battery SOC (MWh)',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, battery_capacity],
            showline=True,
            linecolor=COLORS['primary'],
            linewidth=2,
        ),
        yaxis3=dict(
            title='Battery Power (MW)',
            overlaying='y',
            side='right',
            position=0.85,
            showgrid=False,
            range=[-battery_power*1.2, battery_power*1.2],
            showline=True,
            linecolor=COLORS['primary'],
            linewidth=2,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(df, title="Correlation Matrix"):
    """
    Create a correlation heatmap with retro gaming style
    
    Args:
        df: DataFrame with numeric columns
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    # Calculate correlation matrix
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_cols].corr().round(2)
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale=[COLORS['background'], "#3366CC", COLORS['text'], COLORS['primary']],
        labels=dict(x="Features", y="Features", color="Correlation"),
        text_auto=True
    )
    
    fig.update_layout(
        title=title,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(
            family="VT323",
            size=16,
            color=COLORS['text']
        ),
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels",
            thickness=20,
            tickfont=dict(
                family="VT323",
                size=14,
                color=COLORS['text']
            )
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig