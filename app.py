import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import base64
from PIL import Image
import io
import time

# Set page configuration
st.set_page_config(
    page_title="BATTERY MARKET PREDICTOR",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for retro gaming aesthetic
def load_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono&display=swap');
    
    /* Main container */
    .main {
        background-color: #121212;
        color: #FFD700; /* Golden yellow for text */
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'VT323', monospace !important;
        color: #FF7F50 !important; /* Coral */
        text-shadow: 3px 3px 0px #000000;
        letter-spacing: 2px;
        margin-bottom: 20px !important;
    }
    
    /* Regular text */
    p, div, span, label {
        font-family: 'Space Mono', monospace !important;
        color: #FFD700 !important; /* Golden yellow */
    }
    
    /* Buttons */
    .stButton>button {
        font-family: 'VT323', monospace !important;
        background-color: #FF7F50 !important; /* Coral */
        color: #000000 !important;
        border: 3px solid #FFD700 !important; /* Golden yellow border */
        border-radius: 0px !important; /* Sharp edges */
        padding: 5px 20px !important;
        font-size: 20px !important;
        box-shadow: 5px 5px 0px #000000 !important;
        transition: all 0.1s !important;
    }
    
    .stButton>button:hover {
        transform: translate(2px, 2px) !important;
        box-shadow: 3px 3px 0px #000000 !important;
    }
    
    .stButton>button:active {
        transform: translate(5px, 5px) !important;
        box-shadow: 0px 0px 0px #000000 !important;
    }
    
    /* Sliders and number inputs */
    .stSlider>div>div, .stNumberInput>div>div {
        background-color: #FF7F50 !important; /* Coral */
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #000000 !important;
        border-right: 3px solid #FF7F50 !important; /* Coral border */
    }
    
    /* Metric boxes */
    .css-1xarl3l, .css-1lsmgbg {
        background-color: #000000 !important;
        border: 3px solid #FF7F50 !important; /* Coral border */
        border-radius: 0px !important; /* Sharp edges */
        box-shadow: 5px 5px 0px #FFD700 !important; /* Golden yellow shadow */
        padding: 10px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'VT323', monospace !important;
        background-color: #000000 !important;
        color: #FFD700 !important; /* Golden yellow */
        border: 2px solid #FF7F50 !important; /* Coral border */
        border-radius: 0px !important; /* Sharp edges */
        padding: 5px 20px !important;
        font-size: 18px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF7F50 !important; /* Coral */
        color: #000000 !important;
    }
    
    /* Tables */
    .stDataFrame {
        font-family: 'Space Mono', monospace !important;
        border: 3px solid #FF7F50 !important; /* Coral border */
    }
    
    .stDataFrame th {
        background-color: #FF7F50 !important; /* Coral */
        color: #000000 !important;
        font-family: 'VT323', monospace !important;
        font-size: 18px !important;
        padding: 8px !important;
    }
    
    .stDataFrame td {
        background-color: #000000 !important;
        color: #FFD700 !important; /* Golden yellow */
        font-family: 'Space Mono', monospace !important;
        border: 1px solid #FF7F50 !important; /* Coral border */
        padding: 8px !important;
    }
    
    /* Chart background */
    .js-plotly-plot, .plotly, .plot-container {
        background-color: #000000 !important;
    }
    
    /* Custom containers */
    .pixel-box {
        background-color: #000000;
        border: 3px solid #FF7F50; /* Coral border */
        padding: 20px;
        margin: 10px 0;
        box-shadow: 8px 8px 0px #FFD700; /* Golden yellow shadow */
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #FF7F50 !important; /* Coral */
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Apply the custom CSS
load_css()

# Create a retro pixel art logo
def create_pixel_logo():
    # Create a simple pixel art battery logo
    logo = [
        "                                ",
        "                                ",
        "     ████████████████████      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     █                  █      ",
        "     ████████████████████      ",
        "                                ",
        "                                ",
    ]
    
    # Define colors
    colors = {
        " ": (0, 0, 0, 0),  # Transparent
        "█": (255, 215, 0, 255),  # Golden yellow
    }
    
    # Create a pixel art image
    width = len(logo[0])
    height = len(logo)
    
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = img.load()
    
    for y in range(height):
        for x in range(width):
            if x < len(logo[y]):
                pixels[x, y] = colors[logo[y][x]]
    
    # Scale up the image
    scale = 10
    img = img.resize((width * scale, height * scale), Image.NEAREST)
    
    # Convert to base64 for display
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f'<img src="data:image/png;base64,{img_str}" width="200">'

# Create a blinking text effect
def blinking_text(text, size=24):
    return f"""
    <div style="font-family: 'VT323', monospace; font-size: {size}px; color: #FF7F50; text-align: center; animation: blink 1s infinite;">
        {text}
    </div>
    <style>
        @keyframes blink {{
            0% {{ opacity: 0; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}
    </style>
    """

# Generate sample data for demonstration
def generate_sample_data():
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

# Function to simulate loading with retro game style
def loading_screen():
    st.markdown(blinking_text("LOADING SYSTEM...", 36), unsafe_allow_html=True)
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i in range(101):
        progress_text.markdown(f"<div style='font-family: VT323, monospace; color: #FFD700; font-size: 24px;'>SYSTEM BOOT: {i}%</div>", unsafe_allow_html=True)
        progress_bar.progress(i)
        time.sleep(0.01)
    
    progress_text.markdown(f"<div style='font-family: VT323, monospace; color: #FFD700; font-size: 24px;'>SYSTEM READY!</div>", unsafe_allow_html=True)
    time.sleep(0.5)
    progress_text.empty()
    progress_bar.empty()

# Function for a simple prediction model
def predict_prices(df, features, target, prediction_hours=24):
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

# Main function
def main():
    # Logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(create_pixel_logo(), unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size: 48px; margin-top: 0;'>BATTERY MARKET PREDICTOR</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin-top: 0;'>ANCILLARY SERVICES EDITION</h3>", unsafe_allow_html=True)
    
    # Simulated loading screen at startup
    if 'loaded' not in st.session_state:
        loading_screen()
        st.session_state.loaded = True
    
    # Sidebar navigation
    st.sidebar.markdown("<h2 style='text-align: center;'>CONTROL PANEL</h2>", unsafe_allow_html=True)
    
    # Data options
    st.sidebar.markdown("<div class='pixel-box'><h3 style='margin-top: 0;'>DATA SOURCE</h3></div>", unsafe_allow_html=True)
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Upload CSV", "Use Sample Data"],
        key="data_source"
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
        else:
            st.sidebar.markdown(blinking_text("NO DATA LOADED"), unsafe_allow_html=True)
            if 'data' not in st.session_state:
                st.session_state.data = generate_sample_data()
    else:
        st.sidebar.markdown("<div style='text-align: center;'>🎲 RANDOM DATA GENERATOR 🎲</div>", unsafe_allow_html=True)
        if st.sidebar.button("GENERATE NEW DATA"):
            st.session_state.data = generate_sample_data()
            st.sidebar.success("NEW DATA GENERATED!")
        
        if 'data' not in st.session_state:
            st.session_state.data = generate_sample_data()
    
    # Analysis parameters
    st.sidebar.markdown("<div class='pixel-box'><h3 style='margin-top: 0;'>ANALYSIS SETTINGS</h3></div>", unsafe_allow_html=True)
    
    prediction_target = st.sidebar.selectbox(
        "PREDICT TARGET:",
        ["FCR_price", "aFRR_up_price", "aFRR_down_price"],
        key="prediction_target"
    )
    
    prediction_hours = st.sidebar.slider(
        "PREDICTION HORIZON (HOURS):",
        min_value=1,
        max_value=168,
        value=24,
        step=1,
        key="prediction_hours"
    )
    
    confidence_level = st.sidebar.slider(
        "CONFIDENCE LEVEL (%):",
        min_value=50,
        max_value=99,
        value=95,
        step=1,
        key="confidence_level"
    )
    
    # Battery parameters
    st.sidebar.markdown("<div class='pixel-box'><h3 style='margin-top: 0;'>BATTERY SPECS</h3></div>", unsafe_allow_html=True)
    
    battery_capacity = st.sidebar.number_input(
        "BATTERY CAPACITY (MWh):",
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.1,
        key="battery_capacity"
    )
    
    battery_power = st.sidebar.number_input(
        "BATTERY POWER (MW):",
        min_value=0.1,
        max_value=50.0,
        value=5.0,
        step=0.1,
        key="battery_power"
    )
    
    # Run analysis button
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    run_analysis = st.sidebar.button("► RUN ANALYSIS ◄", key="run_analysis")
    
    # Credits
    st.sidebar.markdown("<br><br><div style='text-align: center; font-size: 12px; opacity: 0.7;'>© 2025 BATTERY MARKET PREDICTOR</div>", unsafe_allow_html=True)
    
    # Main content area with tabs
    tabs = st.tabs(["MARKET DASHBOARD", "PREDICTIONS", "BATTERY ECONOMICS", "DATA EXPLORER"])
    
    with tabs[0]:  # MARKET DASHBOARD
        st.markdown("<h2>MARKET DASHBOARD</h2>", unsafe_allow_html=True)
        
        # Market metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
            st.metric(
                "FCR AVERAGE PRICE",
                f"€{st.session_state.data['FCR_price'].mean():.2f}/MWh",
                f"{st.session_state.data['FCR_price'].mean() - st.session_state.data['FCR_price'].iloc[-24:].mean():.2f}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
            st.metric(
                "aFRR UP AVERAGE PRICE",
                f"€{st.session_state.data['aFRR_up_price'].mean():.2f}/MWh",
                f"{st.session_state.data['aFRR_up_price'].mean() - st.session_state.data['aFRR_up_price'].iloc[-24:].mean():.2f}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
            st.metric(
                "aFRR DOWN AVERAGE PRICE",
                f"€{st.session_state.data['aFRR_down_price'].mean():.2f}/MWh",
                f"{st.session_state.data['aFRR_down_price'].mean() - st.session_state.data['aFRR_down_price'].iloc[-24:].mean():.2f}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Price trends chart
        st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
        st.markdown("<h3>PRICE TRENDS</h3>", unsafe_allow_html=True)
        
        # Get recent data (last 7 days)
        recent_data = st.session_state.data.tail(24*7).copy()
        recent_data['hour'] = recent_data['timestamp'].dt.hour
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['FCR_price'],
            mode='lines',
            name='FCR Price',
            line=dict(color='#FFD700', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['aFRR_up_price'],
            mode='lines',
            name='aFRR Up Price',
            line=dict(color='#FF7F50', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['aFRR_down_price'],
            mode='lines',
            name='aFRR Down Price',
            line=dict(color='#00FF00', width=2)
        ))
        
        fig.update_layout(
            title='Last 7 Days Price Trends',
            xaxis_title='Date',
            yaxis_title='Price (€/MWh)',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(
                family="VT323",
                size=16,
                color="#FFD700"
            ),
            legend=dict(
                font=dict(
                    family="VT323",
                    size=14,
                    color="#FFD700"
                ),
                bgcolor='rgba(0, 0, 0, 0.5)',
                bordercolor='#FF7F50'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='#333333',
                gridwidth=1,
                showline=True,
                linecolor='#FF7F50',
                linewidth=2,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#333333',
                gridwidth=1,
                showline=True,
                linecolor='#FF7F50',
                linewidth=2,
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Price heatmap by hour/day
        st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
        st.markdown("<h3>PRICE HEATMAP</h3>", unsafe_allow_html=True)
        
        # Prepare data for heatmap
        heatmap_data = st.session_state.data.copy()
        heatmap_data['day'] = heatmap_data['timestamp'].dt.day_name()
        heatmap_data['hour'] = heatmap_data['timestamp'].dt.hour
        
        # Calculate average prices by day and hour
        pivot_data = heatmap_data.pivot_table(
            index='day', 
            columns='hour',
            values=prediction_target,
            aggfunc='mean'
        )
        
        # Order days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex(day_order)
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            color_continuous_scale=["black", "#FFD700", "#FF7F50"],
            labels=dict(x="Hour of Day", y="Day of Week", color=prediction_target),
            x=list(range(24)),
            y=day_order
        )
        
        fig.update_layout(
            title=f'{prediction_target} by Day & Hour',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(
                family="VT323",
                size=16,
                color="#FFD700"
            ),
            coloraxis_colorbar=dict(
                title=prediction_target,
                thicknessmode="pixels",
                thickness=20,
                tickfont=dict(
                    family="VT323",
                    size=14,
                    color="#FFD700"
                )
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:  # PREDICTIONS
        st.markdown("<h2>PREDICTIONS</h2>", unsafe_allow_html=True)
        
        if run_analysis:
            with st.spinner("RUNNING PREDICTION MODEL..."):
                # Prepare features for prediction
                features = ['grid_frequency', 'renewable_generation', 'load']
                
                # Run prediction
                predictions = predict_prices(
                    st.session_state.data,
                    features,
                    prediction_target,
                    prediction_hours
                )
                
                # Store predictions
                last_timestamp = st.session_state.data['timestamp'].iloc[-prediction_hours:].reset_index(drop=True)
                pred_df = pd.DataFrame({
                    'timestamp': last_timestamp,
                    'actual': st.session_state.data[prediction_target].iloc[-prediction_hours:].reset_index(drop=True),
                    'predicted': predictions
                })
                
                # Calculate confidence intervals
                std_dev = np.std(pred_df['actual'] - pred_df['predicted'])
                z_score = {
                    90: 1.645,
                    95: 1.96,
                    99: 2.576
                }.get(confidence_level, 1.96)
                
                pred_df['upper_bound'] = pred_df['predicted'] + z_score * std_dev
                pred_df['lower_bound'] = pred_df['predicted'] - z_score * std_dev
                
                # Display prediction metrics
                st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
                st.markdown("<h3>PREDICTION METRICS</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    mae = np.mean(np.abs(pred_df['actual'] - pred_df['predicted']))
                    st.metric("MAE", f"€{mae:.2f}")
                
                with col2:
                    mape = np.mean(np.abs((pred_df['actual'] - pred_df['predicted']) / pred_df['actual'])) * 100
                    st.metric("MAPE", f"{mape:.2f}%")
                
                with col3:
                    max_error = np.max(np.abs(pred_df['actual'] - pred_df['predicted']))
                    st.metric("MAX ERROR", f"€{max_error:.2f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display prediction chart
                st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
                st.markdown("<h3>PRICE PREDICTION</h3>", unsafe_allow_html=True)
                
                fig = go.Figure()
                
                # Actual values
                fig.add_trace(go.Scatter(
                    x=pred_df['timestamp'],
                    y=pred_df['actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#FFD700', width=3),
                    marker=dict(size=8, color='#FFD700')
                ))
                
                # Predicted values
                fig.add_trace(go.Scatter(
                    x=pred_df['timestamp'],
                    y=pred_df['predicted'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#FF7F50', width=3),
                    marker=dict(size=8, color='#FF7F50')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=pred_df['timestamp'].tolist() + pred_df['timestamp'].tolist()[::-1],
                    y=pred_df['upper_bound'].tolist() + pred_df['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 80, 0.2)',
                    line=dict(color='rgba(255, 127, 80, 0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f'{prediction_target} Prediction for next {prediction_hours} hours',
                    xaxis_title='Time',
                    yaxis_title='Price (€/MWh)',
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(
                        family="VT323",
                        size=16,
                        color="#FFD700"
                    ),
                    legend=dict(
                        font=dict(
                            family="VT323",
                            size=14,
                            color="#FFD700"
                        ),
                        bgcolor='rgba(0, 0, 0, 0.5)',
                        bordercolor='#FF7F50'
                    ),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='#333333',
                        gridwidth=1,
                        showline=True,
                        linecolor='#FF7F50',
                        linewidth=2,
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='#333333',
                        gridwidth=1,
                        showline=True,
                        linecolor='#FF7F50',
                        linewidth=2,
                    ),
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display prediction table
                st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
                st.markdown("<h3>PREDICTION DATA</h3>", unsafe_allow_html=True)
                
                display_df = pred_df.copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                display_df = display_df.round(2)
                
                st.dataframe(display_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            st.markdown("<div class='pixel-box' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown(blinking_text("PRESS '► RUN ANALYSIS ◄' BUTTON TO START PREDICTION", 28), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[2]:  # BATTERY ECONOMICS
        st.markdown("<h2>BATTERY ECONOMICS</h2>", unsafe_allow_html=True)
        
        if run_analysis:
            # Battery parameters from sidebar
            batt_capacity = battery_capacity  # MWh
            batt_power = battery_power  # MW
            
            # Create simple battery model
            st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
            st.markdown("<h3>BATTERY SPECS</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CAPACITY", f"{batt_capacity} MWh")
            
            with col2:
                st.metric("POWER", f"{batt_power} MW")
            
            with col3:
                duration = batt_capacity / batt_power
                st.metric("DURATION", f"{duration:.1f} hours")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Calculate potential revenue
            st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
            st.markdown("<h3>REVENUE ESTIMATION</h3>", unsafe_allow_html=True)
            
            # Get price data
            recent_prices = st.session_state.data[prediction_target].iloc[-24*7:].reset_index(drop=True)
            
            # Simplified revenue calculation
            # Assume FCR/aFRR availability payments (not modeling energy payments or activation)
            daily_revenue = np.mean(recent_prices) * batt_power * 24  # € per day
            monthly_revenue = daily_revenue * 30  # € per month
            annual_revenue = daily_revenue * 365  # € per year
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DAILY REVENUE", f"€{daily_revenue:.2f}")
            
            with col2:
                st.metric("MONTHLY REVENUE", f"€{monthly_revenue:.2f}")
            
            with col3:
                st.metric("ANNUAL REVENUE", f"€{annual_revenue:.2f}")
            
            # Simple NPV calculation
            capex = 400000 * batt_power  # € (assuming 400k€/MW)
            opex = 0.02 * capex  # € per year (2% of CAPEX)
            lifetime = 10  # years
            discount_rate = 0.08  # 8%
            
            # Calculate simple NPV
            cash_flows = [-capex]
            for year in range(1, lifetime + 1):
                cash_flows.append(annual_revenue - opex)
            
            npv = np.npv(discount_rate, cash_flows)
            irr = np.irr(cash_flows)
            payback = capex / (annual_revenue - opex)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NPV", f"€{npv:.2f}")
            
            with col2:
                st.metric("IRR", f"{irr*100:.2f}%")
            
            with col3:
                st.metric("PAYBACK PERIOD", f"{payback:.2f} years")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Battery operation strategy
            st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
            st.markdown("<h3>BATTERY OPERATION STRATEGY</h3>", unsafe_allow_html=True)
            
            # Simple battery hourly operation (charge at low price, discharge at high price)
            hourly_prices = recent_prices[-24:].reset_index(drop=True)
            sorted_prices = hourly_prices.sort_values()
            
            charge_hours = sorted_prices.index[:int(duration)]
            discharge_hours = sorted_prices.index[-int(duration):]
            
            # Create operation schedule
            schedule = pd.DataFrame({
                'hour': range(24),
                'price': hourly_prices,
                'operation': 'idle'
            })
            
            for hour in charge_hours:
                schedule.loc[schedule['hour'] == hour, 'operation'] = 'charge'
            
            for hour in discharge_hours:
                schedule.loc[schedule['hour'] == hour, 'operation'] = 'discharge'
            
            # Calculate state of charge
            soc = np.zeros(25)  # 24 hours + initial
            soc[0] = 50  # Start at 50% SOC
            
            for i, row in schedule.iterrows():
                if row['operation'] == 'charge':
                    # Charge at full power if possible
                    energy_change = min(batt_power, batt_capacity - soc[i])
                    soc[i+1] = soc[i] + energy_change
                elif row['operation'] == 'discharge':
                    # Discharge at full power if possible
                    energy_change = min(batt_power, soc[i])
                    soc[i+1] = soc[i] - energy_change
                else:
                    soc[i+1] = soc[i]
            
            # Add SOC to schedule
            schedule['soc'] = soc[1:]
            
            # Display operation strategy
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=schedule['hour'],
                y=schedule['price'],
                mode='lines+markers',
                name=prediction_target,
                line=dict(color='#FFD700', width=2),
                marker=dict(size=8),
                yaxis='y'
            ))
            
            # SOC line
            fig.add_trace(go.Scatter(
                x=schedule['hour'],
                y=schedule['soc'],
                mode='lines+markers',
                name='Battery SOC (MWh)',
                line=dict(color='#FF7F50', width=2),
                marker=dict(size=8),
                yaxis='y2'
            ))
            
            # Operation bars
            colors = {
                'charge': '#00FF00',
                'discharge': '#FF0000',
                'idle': '#808080'
            }
            
            for operation in colors.keys():
                mask = schedule['operation'] == operation
                if mask.any():
                    fig.add_trace(go.Bar(
                        x=schedule.loc[mask, 'hour'],
                        y=[batt_power if operation == 'discharge' else -batt_power if operation == 'charge' else 0] * mask.sum(),
                        name=operation.capitalize(),
                        marker_color=colors[operation],
                        yaxis='y3'
                    ))
            
            fig.update_layout(
                title='24-Hour Battery Operation Strategy',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(
                    family="VT323",
                    size=16,
                    color="#FFD700"
                ),
                legend=dict(
                    font=dict(
                        family="VT323",
                        size=14,
                        color="#FFD700"
                    ),
                    bgcolor='rgba(0, 0, 0, 0.5)',
                    bordercolor='#FF7F50'
                ),
                xaxis=dict(
                    title='Hour',
                    showgrid=True,
                    gridcolor='#333333',
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    gridwidth=1,
                    showline=True,
                    linecolor='#FF7F50',
                    linewidth=2,
                ),
                yaxis=dict(
                    title=prediction_target,
                    showgrid=True,
                    gridcolor='#333333',
                    gridwidth=1,
                    showline=True,
                    linecolor='#FF7F50',
                    linewidth=2,
                ),
                yaxis2=dict(
                    title='Battery SOC (MWh)',
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    range=[0, batt_capacity],
                    showline=True,
                    linecolor='#FF7F50',
                    linewidth=2,
                ),
                yaxis3=dict(
                    title='Battery Power (MW)',
                    overlaying='y',
                    side='right',
                    position=0.85,
                    showgrid=False,
                    range=[-batt_power*1.2, batt_power*1.2],
                    showline=True,
                    linecolor='#FF7F50',
                    linewidth=2,
                ),
                margin=dict(l=10, r=10, t=50, b=10),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display operation schedule
            schedule_display = schedule.copy()
            schedule_display['price'] = schedule_display['price'].round(2)
            schedule_display['soc'] = schedule_display['soc'].round(2)
            
            st.dataframe(schedule_display, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            st.markdown("<div class='pixel-box' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown(blinking_text("PRESS '► RUN ANALYSIS ◄' BUTTON TO START ANALYSIS", 28), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[3]:  # DATA EXPLORER
        st.markdown("<h2>DATA EXPLORER</h2>", unsafe_allow_html=True)
        
        # Raw data table
        st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
        st.markdown("<h3>RAW DATA</h3>", unsafe_allow_html=True)
        
        # Display sample of the data
        st.dataframe(st.session_state.data.tail(100), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data statistics
        st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
        st.markdown("<h3>DATA STATISTICS</h3>", unsafe_allow_html=True)
        
        # Calculate statistics
        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns
        stats_df = st.session_state.data[numeric_cols].describe().round(2)
        
        st.dataframe(stats_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Correlation matrix
        st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
        st.markdown("<h3>CORRELATION MATRIX</h3>", unsafe_allow_html=True)
        
        # Calculate correlation matrix
        corr_matrix = st.session_state.data[numeric_cols].corr().round(2)
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale=["#000000", "#3366CC", "#FFD700", "#FF7F50"],
            labels=dict(x="Features", y="Features", color="Correlation"),
            text_auto=True
        )
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(
                family="VT323",
                size=16,
                color="#FFD700"
            ),
            coloraxis_colorbar=dict(
                title="Correlation",
                thicknessmode="pixels",
                thickness=20,
                tickfont=dict(
                    family="VT323",
                    size=14,
                    color="#FFD700"
                )
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Custom exploration
        st.markdown("<div class='pixel-box'>", unsafe_allow_html=True)
        st.markdown("<h3>CUSTOM EXPLORATION</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-AXIS:", numeric_cols, index=0)
        
        with col2:
            y_axis = st.selectbox("Y-AXIS:", numeric_cols, index=1)
        
        if st.button("GENERATE PLOT"):
            fig = px.scatter(
                st.session_state.data,
                x=x_axis,
                y=y_axis,
                color='battery_soc',
                color_continuous_scale=["#000099", "#00CCFF", "#FFD700", "#FF7F50", "#FF0000"],
                opacity=0.7
            )
            
            fig.update_traces(
                marker=dict(
                    size=10,
                    line=dict(width=2, color='#000000')
                )
            )
            
            fig.update_layout(
                title=f'Scatter Plot: {x_axis} vs {y_axis}',
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(
                    family="VT323",
                    size=16,
                    color="#FFD700"
                ),
                coloraxis_colorbar=dict(
                    title="Battery SOC",
                    thicknessmode="pixels",
                    thickness=20,
                    tickfont=dict(
                        family="VT323",
                        size=14,
                        color="#FFD700"
                    )
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#333333',
                    gridwidth=1,
                    showline=True,
                    linecolor='#FF7F50',
                    linewidth=2,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#333333',
                    gridwidth=1,
                    showline=True,
                    linecolor='#FF7F50',
                    linewidth=2,
                ),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()