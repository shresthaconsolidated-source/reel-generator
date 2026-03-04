import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import asyncio
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips
import moviepy.video.fx.all as vfx
from faster_whisper import WhisperModel
from word2number import w2n
import re

async def generate_tts(text, voice="en-US-GuyNeural", rate="+5%", pitch="-35Hz", output_file="tts_out.mp3"):
    # GuyNeural is a passionate voice. We lower the pitch significantly to make it sound like a deep game announcer.
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def create_outro_frame(damage_text, effort_text, chest_text, output_path):
    img = Image.new('RGB', (1080, 1920), color=(0,0,0))
    d = ImageDraw.Draw(img)
    # Use our newly downloaded cinematic font
    try:
        font_large = ImageFont.truetype("BebasNeue.ttf", 220)
        font_medium = ImageFont.truetype("BebasNeue.ttf", 150)
    except Exception as e:
        print(f"Error loading BebasNeue: {e}")
        font_large = ImageFont.load_default()
        font_medium = font_large
            
    # Calculate text sizes to center them
    def draw_centered_text(y, text, font, fill, shadow_color=(20, 0, 0)):
        bbox = d.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        x = (1080 - w) / 2
        
        # Add drop shadow for cinematic effect
        if shadow_color:
            d.text((x+5, y+5), text, fill=shadow_color, font=font)
        
        d.text((x, y), text, fill=fill, font=font)

    # Draw centered text
    draw_centered_text(500, f"DAMAGE", font_large, (255, 65, 108))
    draw_centered_text(700, f"{damage_text}", font_large, (255, 65, 108))
    
    draw_centered_text(1100, f"EFFORT: {effort_text}", font_medium, (255, 255, 255), (50,50,50))
    draw_centered_text(1300, f"CHEST: {chest_text}", font_medium, (255, 255, 255), (50,50,50))
    
    img.save(output_path)

def process_main_video(input_path, output_temp_path, progress_callback=None):
    # --- 1. Audio Extraction & Transcription ---
    if progress_callback: progress_callback("Extracting Audio...", 0.1)
    
    audio_path = input_path.replace(".mp4", "_audio.wav").replace(".mov", "_audio.wav")
    try:
        video_clip = VideoFileClip(input_path)
        if video_clip.audio is not None:
            video_clip.audio.write_audiofile(audio_path, logger=None)
        video_clip.close()
    except Exception as e:
        print(f"Failed to extract audio: {e}")
        # If no audio or failure, we will just proceed with no numbers detected
        pass

    timestamps = [] # List of tuples: (time_in_seconds, parsed_number_as_int)
    
    if os.path.exists(audio_path):
        if progress_callback: progress_callback("Transcribing Pushup Counts (Whisper AI)...", 0.2)
        # Using tiny/base model for speed. It runs locally.
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        
        # Word-level timestamps to know exactly when a number was spoken
        segments, info = model.transcribe(audio_path, word_timestamps=True, language="en")
        
        for segment in segments:
            for word in segment.words:
                clean_word = re.sub(r'[^a-zA-Z0-9]', '', word.word).strip().lower()
                
                # Check if it's a number word or a digit
                try:
                    num = w2n.word_to_num(clean_word)
                    timestamps.append((word.start, num))
                    print(f"Detected number: {num} at {word.start}s")
                except ValueError:
                    # Not a parsable number word, skip
                    pass
        
        try: os.remove(audio_path)
        except: pass

    # --- 2. Video Frame Processing ---
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp_path, fourcc, fps, (width, height))
    
    # Load face model (Pose model removed as per new architecture)
    face_model = YOLO('yolov8n-face.pt')
    
    frame_idx = 0
    max_count = 0
    
    # We will "flash" the number on screen for 1 second after it is spoken
    flash_duration_frames = int(fps * 1.0)
    current_flash_number = None
    flash_frames_remaining = 0
    
    # Sort timestamps chronologically just in case
    timestamps.sort(key=lambda x: x[0])
    next_timestamp_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        current_time_sec = frame_idx / fps
        
        if progress_callback and total_frames > 0 and frame_idx % 10 == 0:
            progress_callback(f"Rendering Video Visuals (Frame {frame_idx}/{total_frames})", 0.3 + (0.6 * (frame_idx / total_frames)))
            
        # Check if we should start a new flash based on whisper timestamps
        while next_timestamp_idx < len(timestamps) and current_time_sec >= timestamps[next_timestamp_idx][0]:
            current_flash_number = timestamps[next_timestamp_idx][1]
            max_count = max(max_count, current_flash_number) # Keep track of max pushups spoken
            flash_frames_remaining = flash_duration_frames
            next_timestamp_idx += 1
            
        # --- Face Blurring using YOLO ---
        face_results = face_model(frame, verbose=False)
        for result in face_results:
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    
                    # Add padding
                    w = x2 - x1
                    h = y2 - y1
                    pad_w = int(w * 0.2)
                    pad_h = int(h * 0.2)
                    
                    x_min = max(0, x1 - pad_w)
                    y_min = max(0, y1 - pad_h)
                    x_max = min(width, x2 + pad_w)
                    y_max = min(height, y2 + pad_h)
                    
                    face_roi = frame[y_min:y_max, x_min:x_max]
                    if face_roi.size > 0:
                        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                        frame[y_min:y_max, x_min:x_max] = blurred_face
        
        # --- Number Screen Overlay ---
        if flash_frames_remaining > 0 and current_flash_number is not None:
            # Draw giant red number in the top left
            cv2.putText(frame, str(current_flash_number), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 15, cv2.LINE_AA)
            cv2.putText(frame, str(current_flash_number), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 5, cv2.LINE_AA)
            flash_frames_remaining -= 1
        
        out.write(frame)
        
    cap.release()
    out.release()
    return max_count

def generate_reels_pipeline(main_video_path, screenshot_path, follower_count, damage_text, effort_text, chest_text, output_path, start_fast, end_fast, speed_factor=2.0, progress_callback=None):
    temp_files = []
    
    def get_temp_file(suffix):
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        temp_files.append(path)
        return path
        
    try:
        # 1. CREATE INTRO
        if progress_callback: progress_callback("Generating AI Voice Intro...", 0.0)
        intro_audio_path = get_temp_file('.mp3')
        
        if follower_count == 0:
            intro_text = "1 new follower equals 1 pushup. We got 0 new followers today. So today... we do 0 pushups."
        elif follower_count == 1:
            intro_text = "1 new follower equals 1 pushup. We got 1 new follower, so today we do 1 pushup."
        else:
            intro_text = f"1 new follower equals 1 pushup. We got {follower_count} new followers, so today we do {follower_count} pushups."
            
        asyncio.run(generate_tts(intro_text, output_file=intro_audio_path))
        
        intro_audio = AudioFileClip(intro_audio_path)
        intro_clip = ImageClip(screenshot_path).set_duration(intro_audio.duration + 0.5)
        # Resize to typical reels ratio (keep aspect, pad black) to ensure consistency, but if screenshot is different size, it might look odd.
        # MoviePy's resize will handle it. We will resize all clips to standard 1080x1920.
        intro_clip = intro_clip.resize(height=1920).margin(color=(0,0,0)).set_audio(intro_audio)
        if intro_clip.w > 1080: intro_clip = intro_clip.crop(x_center=intro_clip.w/2, width=1080)
        
        # 2. CREATE MAIN PUSHUP VIDEO
        main_processed_path = get_temp_file('.mp4')
        pushup_count = process_main_video(main_video_path, main_processed_path, progress_callback)
        
        if progress_callback: progress_callback("Stitching and Enhancing Video Speed...", 0.99)
        main_clip = VideoFileClip(main_processed_path)
        main_clip = main_clip.resize(height=1920)
        if main_clip.w > 1080: main_clip = main_clip.crop(x_center=main_clip.w/2, width=1080)
        
        # Apply fast forward styling to mid section
        duration = main_clip.duration
        start_fast = max(0, start_fast)
        end_fast = min(duration, end_fast)
        
        main_concat_clips = []
        if start_fast > 0:
            main_concat_clips.append(main_clip.subclip(0, start_fast))
        if end_fast > start_fast:
            main_concat_clips.append(main_clip.subclip(start_fast, end_fast).fx(vfx.speedx, speed_factor))
        if end_fast < duration:
            main_concat_clips.append(main_clip.subclip(end_fast, duration))
            
        if main_concat_clips:
            final_main_clip = concatenate_videoclips(main_concat_clips)
        else:
            final_main_clip = main_clip
            
        # 3. CREATE OUTRO
        outro_audio_path = get_temp_file('.mp3')
        outro_text = f"Damage {damage_text}. Effort {effort_text}. Chest {chest_text}."
        asyncio.run(generate_tts(outro_text, output_file=outro_audio_path))
        
        outro_img_path = get_temp_file('.png')
        create_outro_frame(damage_text, effort_text, chest_text, outro_img_path)
        
        outro_audio = AudioFileClip(outro_audio_path)
        outro_clip = ImageClip(outro_img_path).set_duration(outro_audio.duration + 0.5).set_audio(outro_audio)
        outro_clip = outro_clip.resize(height=1920)
        
        # 4. ASSEMBLE ALL
        # We need to ensure all clips have the exact same size. Let's force everything to 1080x1920 resolution.
        # Alternatively, we just concatenate and let moviepy handle it via method='compose'
        final_video = concatenate_videoclips([intro_clip, final_main_clip, outro_clip], method="compose")
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30, logger=None)
        
        # Close everything
        intro_audio.close()
        intro_clip.close()
        main_clip.close()
        final_main_clip.close()
        outro_audio.close()
        outro_clip.close()
        final_video.close()
        
    finally:
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
                
    return pushup_count
