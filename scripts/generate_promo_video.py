import os
import subprocess
import json
import sys
from PIL import Image, ImageDraw, ImageFont

# Path configuration (surchargeable sans chemin personnel codé en dur)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIR = os.environ.get("VIDEO_AGENT_PROMO_CLIPS", os.path.join(PROJECT_ROOT, "promo_clips"))
OUTPUT_DIR = os.environ.get(
    "VIDEO_AGENT_PROMO_OUTPUT", os.path.join(PROJECT_ROOT, "outputs", "promo_build")
)
FINAL_OUTPUT = os.path.join(CLIPS_DIR, "video_abajad_ai.mp4")
LOGO_PATH = os.environ.get("VIDEO_AGENT_PROMO_LOGO", os.path.join(PROJECT_ROOT, "assets", "logo.png"))
EDGE_TTS_BIN = os.environ.get("EDGE_TTS_BIN", "edge-tts")
BG_MUSIC_PATH = os.environ.get(
    "VIDEO_AGENT_PROMO_MUSIC", os.path.join(PROJECT_ROOT, "assets", "bg_music.mp3")
)

# Voice configuration
VOICE = "fr-FR-HenriNeural"

# Video dimensions & framerate
WIDTH = 1280
HEIGHT = 720
FPS = 24
CLIP_DURATION = 10.005  # exactly matching original clips
END_CARD_DURATION = 4.0

# Clip texts and voiceover script
CLIPS_DATA = [
    {
        "file": "Clip1.mp4",
        "text": ["Abajad AI", "Des solutions IA pour automatiser votre business"],
        "speech": "Votre entreprise perd du temps avec les appels, la comptabilité et les tâches répétitives."
    },
    {
        "file": "Clip2.mp4",
        "text": ["Abajad AI", "Des solutions IA pour automatiser votre business"],
        "speech": "Abajad AI transforme ces défis en solutions intelligentes, automatisées et disponibles vingt-quatre heures sur vingt-quatre."
    },
    {
        "file": "Clip3.mp4",
        "text": ["Nemra 06", "Assistant téléphonique IA"],
        "speech": "Avec Nemra 06, vos appels sont traités automatiquement, en arabe, en français et en anglais."
    },
    {
        "file": "Clip4.mp4",
        "text": ["ComptaHub", "Comptabilité automatisée"],
        "speech": "Avec ComptaHub, votre comptabilité devient plus simple, plus rapide et mieux organisée."
    },
    {
        "file": "Clip5.mp4",
        "text": ["NHLA", "ERP intelligent"],
        "speech": "Avec NHLA, vous pilotez votre entreprise : ressources humaines, stock, finance et production, depuis un seul système intelligent."
    },
    {
        "file": "Clip6.mp4",
        "text": ["Abajad AI", "Connectez vos outils, automatisez vos opérations et décidez mieux"],
        "speech": "Abajad AI connecte vos outils, automatise vos opérations et vous aide à prendre de meilleures décisions."
    },
    {
        "file": "Clip7.mp4",
        "text": ["Transformez votre business avec l’IA", "Abajad AI"],
        "speech": "Abajad AI. Transformez votre business avec l’intelligence artificielle."
    }
]

def run_command(cmd, desc=""):
    if desc:
        print(f"--- {desc} ---")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error in: {desc}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result

def create_text_overlay(lines, output_path):
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Try loading Arial
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 36)
        font_sub = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        try:
            font_title = ImageFont.truetype("arial.ttf", 36)
            font_sub = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font_title = ImageFont.load_default()
            font_sub = ImageFont.load_default()
            
    box_width = 1000
    box_height = 140
    box_x1 = (WIDTH - box_width) // 2
    box_y1 = HEIGHT - box_height - 50
    box_x2 = box_x1 + box_width
    box_y2 = box_y1 + box_height
    
    # Semi-transparent dark background
    draw.rounded_rectangle([box_x1, box_y1, box_x2, box_y2], radius=15, fill=(0, 0, 0, 160))
    
    if len(lines) == 1:
        text = lines[0]
        w = draw.textlength(text, font=font_title)
        h = 40
        tx = box_x1 + (box_width - w) // 2
        ty = box_y1 + (box_height - h) // 2
        draw.text((tx, ty), text, font=font_title, fill=(255, 255, 255, 255))
    elif len(lines) >= 2:
        title = lines[0]
        sub = lines[1]
        
        tw = draw.textlength(title, font=font_title)
        sw = draw.textlength(sub, font=font_sub)
        
        th = 40
        sh = 28
        gap = 10
        total_text_h = th + sh + gap
        
        tx = box_x1 + (box_width - tw) // 2
        ty = box_y1 + (box_height - total_text_h) // 2
        
        sx = box_x1 + (box_width - sw) // 2
        sy = ty + th + gap
        
        # Gold for Title, White for subtitle
        draw.text((tx, ty), title, font=font_title, fill=(218, 165, 32, 255))
        draw.text((sx, sy), sub, font=font_sub, fill=(255, 255, 255, 255))
        
    img.save(output_path, "PNG")
    print(f"Saved overlay: {output_path}")

def create_end_card(logo_path, output_path):
    img = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
    
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        logo_w, logo_h = logo.size
        # upscale by 4x using Lanczos
        new_logo_w = logo_w * 4
        new_logo_h = logo_h * 4
        logo_resized = logo.resize((new_logo_w, new_logo_h), Image.Resampling.LANCZOS)
        
        logo_x = (WIDTH - new_logo_w) // 2
        logo_y = HEIGHT // 2 - new_logo_h - 20
        
        img.paste(logo_resized, (logo_x, logo_y), logo_resized if logo_resized.mode == 'RGBA' else None)
    else:
        print("Warning: logo image not found!")
        
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 48)
        font_sub = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        
    title_text = "Abajad AI"
    sub_text = "Des solutions IA pour automatiser votre business"
    
    tw = draw.textlength(title_text, font=font_title)
    tx = (WIDTH - tw) // 2
    ty = HEIGHT // 2 + 40
    
    sw = draw.textlength(sub_text, font=font_sub)
    sx = (WIDTH - sw) // 2
    sy = ty + 60
    
    draw.text((tx, ty), title_text, font=font_title, fill=(50, 50, 50, 255))
    draw.text((sx, sy), sub_text, font=font_sub, fill=(100, 100, 100, 255))
    
    img.convert("RGB").save(output_path, "JPEG", quality=95)
    print(f"Saved end card image: {output_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Generate voiceover tracks using edge-tts
    print("\n=== STEP 1: Generating Voiceover Tracks ===")
    tts_files = []
    for i, clip in enumerate(CLIPS_DATA):
        tts_file = os.path.join(OUTPUT_DIR, f"tts_{i+1}.mp3")
        tts_files.append(tts_file)
        
        # Call edge-tts
        cmd = [
            EDGE_TTS_BIN,
            "--text", clip["speech"],
            "--write-media", tts_file,
            "--voice", VOICE
        ]
        run_command(cmd, f"Generating voiceover for clip {i+1}")
        
    # Step 2: Generate text overlay PNGs
    print("\n=== STEP 2: Creating Text Overlays ===")
    overlay_files = []
    for i, clip in enumerate(CLIPS_DATA):
        overlay_file = os.path.join(OUTPUT_DIR, f"overlay_{i+1}.png")
        overlay_files.append(overlay_file)
        create_text_overlay(clip["text"], overlay_file)
        
    # Step 3: Create logo end card
    print("\n=== STEP 3: Creating End Card ===")
    end_card_img = os.path.join(OUTPUT_DIR, "logo_endcard.jpg")
    create_end_card(LOGO_PATH, end_card_img)
    
    end_card_video = os.path.join(OUTPUT_DIR, "logo_endcard.mp4")
    # Convert image to 4s silent video
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", end_card_img,
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(END_CARD_DURATION),
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        end_card_video
    ]
    run_command(cmd, "Converting end card to video")
    
    # Step 4: Process and compile each clip with overlay and TTS
    print("\n=== STEP 4: Processing Individual Clips ===")
    processed_clips = []
    for i, clip in enumerate(CLIPS_DATA):
        input_video = os.path.join(CLIPS_DIR, clip["file"])
        output_video = os.path.join(OUTPUT_DIR, f"processed_{i+1}.mp4")
        processed_clips.append(output_video)
        
        overlay_file = overlay_files[i]
        tts_file = tts_files[i]
        
        # Determine filter complex: Apply delogo for Clip3.mp4 to hide the vulgar word
        if clip["file"] == "Clip3.mp4":
            filter_complex = "[0:v]delogo=x=195:y=520:w=120:h=80[clean_v];[clean_v][1:v]overlay=0:0[v];[2:a][3:a]amix=inputs=2:duration=first:dropout_transition=0[a]"
            desc_detail = f"Processing clip {i+1} (with delogo filter)"
        else:
            filter_complex = "[0:v][1:v]overlay=0:0[v];[2:a][3:a]amix=inputs=2:duration=first:dropout_transition=0[a]"
            desc_detail = f"Processing clip {i+1}"

        # Overlay PNG and merge TTS audio (padded to exactly 10.005 seconds with silence)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", overlay_file,
            "-i", tts_file,
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-t", str(CLIP_DURATION),
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            output_video
        ]
        run_command(cmd, desc_detail)
        
    # Step 5: Concat all clips (7 clips + end card)
    print("\n=== STEP 5: Concatenating Clips ===")
    concat_list_file = os.path.join(OUTPUT_DIR, "concat_list.txt")
    with open(concat_list_file, "w") as f:
        for clip in processed_clips:
            f.write(f"file '{os.path.basename(clip)}'\n")
        f.write(f"file '{os.path.basename(end_card_video)}'\n")
        
    combined_temp = os.path.join(OUTPUT_DIR, "combined_temp.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_file,
        "-c", "copy",
        combined_temp
    ]
    run_command(cmd, "Concatenating segments")
    
    # Step 6: Add background music & apply transitions (fades)
    print("\n=== STEP 6: Mixing Background Music and Applying Transitions ===")
    total_duration = CLIP_DURATION * len(CLIPS_DATA) + END_CARD_DURATION # 74.035
    
    cmd = [
        "ffmpeg", "-y",
        "-i", combined_temp,
        "-i", BG_MUSIC_PATH,
        "-filter_complex", 
        f"[0:v]fade=t=in:st=0:d=1,fade=t=out:st={total_duration-2.0}:d=2[v];"
        f"[1:a]volume=0.07,afade=t=out:st={total_duration-4.0}:d=4[bg];"
        f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0,afade=t=in:st=0:d=1[a]",
        "-map", "[v]", "-map", "[a]",
        "-t", str(total_duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        FINAL_OUTPUT
    ]
    run_command(cmd, "Final mix and transitions")
    
    print("\n==========================================")
    print("PROMOTION VIDEO COMPILATION SUCCESSFUL!")
    print(f"Final output video saved at: {FINAL_OUTPUT}")
    print("==========================================")

if __name__ == "__main__":
    main()
