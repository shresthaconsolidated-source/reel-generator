import streamlit as st
import tempfile
import os
from processor import generate_reels_pipeline
from PIL import Image

st.set_page_config(page_title="Reel Builder", page_icon="⚡", layout="centered")

def inject_premium_css():
    st.markdown("""
        <style>
        /* Modern Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif !important;
            background-color: #0b0c10;
            color: #c5c6c7;
        }
        
        /* Hide Default Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Premium Title Gradient */
        .premium-title {
            background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800;
            text-align: center;
            margin-bottom: -15px;
            padding-top: 2rem;
        }
        
        .sub-title {
            text-align: center;
            color: #8f929a;
            font-weight: 300;
            margin-bottom: 3rem;
        }

        /* Glassmorphism Containers */
        .glass-container {
            background: rgba(31, 33, 40, 0.45);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        
        .section-header {
            color: #ffffff;
            font-weight: 600;
            font-size: 1.3rem;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 10px;
        }

        /* Input Overrides */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: rgba(0, 0, 0, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #fff !important;
            border-radius: 8px !important;
        }
        
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
            border: 1px solid #ff416c !important;
            box-shadow: 0 0 10px rgba(255, 65, 108, 0.3) !important;
        }
        
        /* Premium Button */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 65, 108, 0.6);
            color: white;
            border: none;
        }
        
        .stFileUploader>div>div {
            background-color: rgba(0, 0, 0, 0.2) !important;
            border: 1px dashed rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
        }
        </style>
    """, unsafe_allow_html=True)

inject_premium_css()

st.markdown('<h1 class="premium-title">AI Reel Builder</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automate your TikTok/Reels pushup challenges in seconds.</p>', unsafe_allow_html=True)

st.markdown('<div class="glass-container"><div class="section-header">1. Challenge Details</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    follower_count = st.number_input("Follower Count", min_value=0, value=20, step=1)
    damage_text = st.text_input("Damage Rating", value="20")
with col_b:
    effort_text = st.text_input("Effort Rating", value="Absolutely Brutal")
    chest_text = st.text_input("Chest Status", value="DESTROYED")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-container"><div class="section-header">2. Media Uploads</div>', unsafe_allow_html=True)
screenshot_file = st.file_uploader("Upload Follower Screenshot", type=["png", "jpg", "jpeg"])
video_file = st.file_uploader("Upload Pushup Video", type=["mp4", "mov", "avi"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-container"><div class="section-header">3. Timeline & Speed</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    start_fast = st.number_input("Fast Forward Start (sec)", min_value=0.0, value=2.0, step=0.5)
with col2:
    end_fast = st.number_input("Fast Forward End (sec)", min_value=0.0, value=10.0, step=0.5)
    
speed_factor = st.slider("Fast Forward Speed Multiplier", min_value=1.5, max_value=4.0, value=3.0, step=0.5)
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Generate My Reel"):
    if not screenshot_file or not video_file:
        st.error("Please upload both the screenshot and the pushup video first.")
    else:
        with st.spinner("Generating Reel..."):
            progress_bar = st.progress(0.0, text="Preparing video processing pipeline...")
            
            def progress_callback(text, percent):
                progress_bar.progress(percent, text=text)
                
            fd1, temp_screenshot = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd1, 'wb') as f: f.write(screenshot_file.read())
                
            fd2, temp_video = tempfile.mkstemp(suffix=".mp4")
            with os.fdopen(fd2, 'wb') as f: f.write(video_file.read())
            
            fd_out, temp_out = tempfile.mkstemp(suffix=".mp4")
            os.close(fd_out)
            
            try:
                pushup_count = generate_reels_pipeline(
                    main_video_path=temp_video,
                    screenshot_path=temp_screenshot,
                    follower_count=follower_count,
                    damage_text=damage_text,
                    effort_text=effort_text,
                    chest_text=chest_text,
                    output_path=temp_out,
                    start_fast=start_fast,
                    end_fast=end_fast,
                    speed_factor=speed_factor,
                    progress_callback=progress_callback
                )
                
                progress_bar.empty()
                st.success(f"Video created! Successfully logged {pushup_count} pushups in the vision system.")
                
                with open(temp_out, 'rb') as f:
                    video_bytes = f.read()
                    st.video(video_bytes)
                    st.download_button(
                        label="Download Final Reel",
                        data=video_bytes,
                        file_name="challenge_reel.mp4",
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            finally:
                if os.path.exists(temp_screenshot): os.remove(temp_screenshot)
                if os.path.exists(temp_video): os.remove(temp_video)
                if os.path.exists(temp_out): os.remove(temp_out)
