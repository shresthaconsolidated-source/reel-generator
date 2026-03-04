import streamlit as st
import tempfile
import os
from processor import generate_reels_pipeline
from PIL import Image

st.set_page_config(page_title="Automated Reels Generator", layout="centered")

st.title("Automated Reels Generator")
st.markdown("Create a fully automated TikTok/Reels pushup challenge video. We generate the voiceovers, intro screen, and outro damage report!")

st.subheader("1. Setup your Challenge")
follower_count = st.number_input("Follower Count (e.g. 20)", min_value=1, value=20, step=1)
effort_text = st.text_input("Effort Rating", value="Absolutely Brutal")

st.subheader("2. Upload Files")
screenshot_file = st.file_uploader("Upload Follower Screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
video_file = st.file_uploader("Upload Pushup Video (MP4/MOV)", type=["mp4", "mov", "avi"])

st.subheader("3. Video Edits")
col1, col2 = st.columns(2)
with col1:
    start_fast = st.number_input("Fast Forward Start (seconds)", min_value=0.0, value=2.0, step=0.5)
with col2:
    end_fast = st.number_input("Fast Forward End (seconds)", min_value=0.0, value=10.0, step=0.5)
    
speed_factor = st.slider("Fast Forward Speed", min_value=1.5, max_value=4.0, value=3.0, step=0.5)

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
                    effort_text=effort_text,
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
