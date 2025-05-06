import streamlit as st
import base64
from PIL import Image
import io

def load_css():
    """Load custom CSS for retro gaming aesthetic"""
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
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def create_pixel_logo():
    """Create a simple pixel art battery logo"""
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

def blinking_text(text, size=24):
    """Create blinking text effect for retro feel"""
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

def loading_screen():
    """Display a retro-style loading screen"""
    st.markdown(blinking_text("LOADING SYSTEM...", 36), unsafe_allow_html=True)
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    import time
    for i in range(101):
        progress_text.markdown(f"<div style='font-family: VT323, monospace; color: #FFD700; font-size: 24px;'>SYSTEM BOOT: {i}%</div>", unsafe_allow_html=True)
        progress_bar.progress(i)
        time.sleep(0.01)
    
    progress_text.markdown(f"<div style='font-family: VT323, monospace; color: #FFD700; font-size: 24px;'>SYSTEM READY!</div>", unsafe_allow_html=True)
    time.sleep(0.5)
    progress_text.empty()
    progress_bar.empty()