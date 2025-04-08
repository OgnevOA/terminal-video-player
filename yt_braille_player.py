#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import sys
import subprocess
import argparse
import signal
import math
import threading
import io
import tempfile
import traceback # For better error printing

# --- Attempt to import audio libraries ---
AUDIO_ENABLED = True # Assume true initially
try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    print("pydub imported successfully.")
except ImportError:
    print("Warning: 'pydub' not found. Audio playback and effects will be disabled.", file=sys.stderr)
    print("Install it using: pip install pydub", file=sys.stderr)
    AudioSegment = None
    AUDIO_ENABLED = False

try:
    import sounddevice as sd
    print("sounddevice imported successfully.")
except ImportError:
    print("Warning: 'sounddevice' not found. Audio playback will be disabled.", file=sys.stderr)
    print("Install it using: pip install sounddevice", file=sys.stderr)
    sd = None
    AUDIO_ENABLED = False

# Requests is still needed for the old download method fallback, but primary download uses yt-dlp
try:
    import requests
    print("requests imported successfully (needed for URL check).")
except ImportError:
    print("Warning: 'requests' not found. Basic URL checks might fail.", file=sys.stderr)
    # Don't disable AUDIO_ENABLED here, as yt-dlp download doesn't need it.
    requests = None

# --- Configuration & Constants ---
ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]
DEFAULT_THRESHOLD = 120
BRAILLE_BASE_CODEPOINT = 0x2800
DOT_MAP = {
    (0, 0): 1 << 0, (1, 0): 1 << 1, (2, 0): 1 << 2, (3, 0): 1 << 6,
    (0, 1): 1 << 3, (1, 1): 1 << 4, (2, 1): 1 << 5, (3, 1): 1 << 7
}
WEBCAM_IDENTIFIER = "webcam"
AUDIO_DOWNLOAD_TIMEOUT = 600 # Seconds (10 minutes)

# --- Global flag for stopping audio ---
stop_audio_flag = threading.Event()

# --- Helper Functions ---
def get_terminal_dimen():
    try: cols, lines = os.get_terminal_size(); return cols, lines
    except OSError: return 80, 24

# Add near the top imports
import uuid
import traceback # Already there but good practice

def get_video_url_and_download_audio(youtube_url, attempt_audio_download=True):
    """
    Gets video URL (<=480p).
    If attempt_audio_download is True, downloads best audio using yt-dlp
    to a temporary file path (created by this function) and returns its path.
    Returns: (video_url, audio_file_path or None)
    """
    video_url = None
    audio_file_path = None # Initialize path variable

    # --- Get Video URL ---
    print("Fetching video stream URL (max 480p)...")
    try:
        v_format = 'bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]'
        v_command = ['yt-dlp', '-f', v_format, '--get-url', youtube_url]
        v_result = subprocess.run(v_command, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
        potential_urls = v_result.stdout.strip().split('\n')
        video_url = next((url for url in potential_urls if url.startswith('http')), None)
        if video_url:
            print("Video URL obtained.")
        else:
            print("Warning: Could not extract video URL.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running yt-dlp for video URL: {e}\nstderr: {e.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"Error getting video stream URL: {e}", file=sys.stderr)

    if not video_url:
        print("Could not get video URL, cannot proceed.", file=sys.stderr)
        return None, None # Can't continue without video

    # --- Download Audio using yt-dlp ---
    if not attempt_audio_download:
        print("Audio download skipped.")
        return video_url, None

    print("Attempting to download audio using yt-dlp...")
    # --- Generate unique path without creating the file ---
    try:
        temp_dir = tempfile.gettempdir()
        temp_filename = f"yt_audio_{uuid.uuid4()}.audio" # Use UUID
        audio_file_path = os.path.join(temp_dir, temp_filename)
        if os.path.exists(audio_file_path):
            print(f"Warning: Generated temp path exists? {audio_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error generating temporary file path: {e}", file=sys.stderr)
        return video_url, None
    # --- END CHANGE AREA ---

    # Now 'audio_file_path' is just a path string, the file doesn't exist yet.

    try:
        a_format = 'bestaudio/best'
        downloader_args = []
        try: # Check for aria2c
            if os.name == 'nt':
                subprocess.run(['where', 'aria2c'], check=True, capture_output=True)
            else:
                subprocess.run(['which', 'aria2c'], check=True, capture_output=True)
            print("Found aria2c, using it as downloader.")
            downloader_args = ['--downloader', 'aria2c']
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("aria2c not found, using default yt-dlp downloader.")

        a_command = [
            'yt-dlp',
            '-f', a_format,
            '--progress', # Show progress
            *downloader_args, # Add downloader args if found
            '-o', audio_file_path, # Tell yt-dlp to create and write to this path
            youtube_url
        ]
        print(f"Running yt-dlp to download audio to: {audio_file_path}")
        # yt-dlp will now create the file at audio_file_path
        dl_result = subprocess.run(a_command, check=True, timeout=AUDIO_DOWNLOAD_TIMEOUT)
        print("\nyt-dlp audio download finished successfully.")
        # Verify file exists and has size now
        if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
            print("Error: yt-dlp reported success, but downloaded file is missing or empty.", file=sys.stderr)
            if os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path) # Clean up empty file
                except OSError: pass
            return video_url, None
        return video_url, audio_file_path # Return the path to the (now existing) file

    except subprocess.TimeoutExpired:
        print("\nError: yt-dlp audio download timed out.", file=sys.stderr)
        if audio_file_path and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except OSError: pass
        return video_url, None
    except subprocess.CalledProcessError as e:
        print(f"\nError running yt-dlp for audio download (return code {e.returncode}).", file=sys.stderr)
        if audio_file_path and os.path.exists(audio_file_path):
            print(f"yt-dlp failed, removing potentially incomplete file: {audio_file_path}")
            try:
                os.remove(audio_file_path)
            except OSError: pass
        return video_url, None
    except Exception as e:
        print(f"\nAn unexpected error occurred during yt-dlp audio download: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if audio_file_path and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except OSError: pass
        return video_url, None


def bgr_to_rgb_ansi(bgr_color):
    b = max(0, min(255, int(bgr_color[0]))); g = max(0, min(255, int(bgr_color[1]))); r = max(0, min(255, int(bgr_color[2])))
    return f"\033[38;2;{r};{g};{b}m"

# --- Audio Processing and Playback ---
def apply_8bit_effect(audio_segment):
    if not AudioSegment: return None, 0
    try:
        target_sample_rate = 11025
        processed_segment = audio_segment.set_frame_rate(target_sample_rate)
        processed_segment = processed_segment.set_sample_width(1)
        processed_segment = processed_segment.normalize()
        print(f"Applied 8-bit effect (Rate: {target_sample_rate} Hz, Width: 8-bit)")
        return processed_segment, target_sample_rate
    except Exception as e:
        print(f"Error applying 8-bit effect: {e}", file=sys.stderr)
        return audio_segment, audio_segment.frame_rate

def play_audio_thread(audio_segment, sample_rate):
    """Plays the given pydub AudioSegment in a separate thread."""
    if not sd or not audio_segment:
        print("Audio playback prerequisites not met in thread.")
        return

    try:
        num_channels = audio_segment.channels
        sample_width = audio_segment.sample_width

        if sample_width == 1: dtype = np.uint8
        elif sample_width == 2: dtype = np.int16
        elif sample_width == 4: dtype = np.int32
        else: print(f"Error: Unsupported sample width: {sample_width}", file=sys.stderr); return

        samples_raw = np.frombuffer(audio_segment.raw_data, dtype=dtype)
        if num_channels > 1:
            try: samples_reshaped = samples_raw.reshape(-1, num_channels)
            except ValueError as e:
                 print(f"Error reshaping audio data: {e}", file=sys.stderr); print("Attempting mono...", file=sys.stderr)
                 samples_reshaped = samples_raw; num_channels = 1
        elif num_channels == 1: samples_reshaped = samples_raw
        else: print(f"Error: Invalid channel count ({num_channels}).", file=sys.stderr); return

        if samples_reshaped.dtype == np.uint8: samples_float = (samples_reshaped.astype(np.float32) / 127.5) - 1.0
        elif samples_reshaped.dtype == np.int16: samples_float = samples_reshaped.astype(np.float32) / (2**15)
        elif samples_reshaped.dtype == np.int32: samples_float = samples_reshaped.astype(np.float32) / (2**31)
        else: print(f"Error: Unexpected dtype {samples_reshaped.dtype}", file=sys.stderr); return
        samples_float = samples_float.astype(np.float32)

        inferred_channels = samples_float.shape[1] if samples_float.ndim > 1 else 1
        print(f"Starting audio playback (Duration: {len(audio_segment)/1000:.2f}s, Channels inferred: {inferred_channels}, Rate: {sample_rate}Hz)...")

        # Play without callback or explicit channels
        sd.play(samples_float, samplerate=sample_rate, blocking=False)

        playback_start_time = time.monotonic()
        while sd.get_stream().active and not stop_audio_flag.is_set():
             time.sleep(0.1)

        if stop_audio_flag.is_set():
            print("Audio stop signaled.")
            if sd.get_stream().active: print("Stopping active audio stream due to flag."); sd.stop()
        elif not sd.get_stream().active: print("Audio stream finished naturally.")

        print("Audio playback thread finishing.")

    except sd.PortAudioError as pae: print(f"\nPortAudioError: {pae}", file=sys.stderr)
    except Exception as e: print(f"\nError during audio playback thread: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
    finally:
        try:
            if sd.get_stream().active: print("Stopping audio stream in finally block."); sd.stop()
        except Exception: pass


# --- Rendering Logic ---
def asciify_pixel(pixel_bgr):
    brightness = sum(pixel_bgr) / 3
    norm_brightness = brightness / 255.0
    index = min(len(ASCII_CHARS) - 1, int(norm_brightness * len(ASCII_CHARS)))
    return ASCII_CHARS[len(ASCII_CHARS) - 1 - index]

def render_frame_ascii(frame, term_cols, term_rows):
    if term_cols <= 0 or term_rows <= 0: return ""
    height, width, _ = frame.shape
    if height == 0 or width == 0: return ""
    aspect_ratio = width / height
    new_width = term_cols
    new_height = max(1, int((new_width / aspect_ratio) / 2))
    new_height = min(term_rows -1, new_height)
    new_width = max(1, int(new_height * aspect_ratio * 2))
    new_width = min(term_cols, new_width)
    if new_width <= 0 or new_height <= 0: return ""
    try: resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e: print(f"\nError resizing frame (ASCII): {e}", file=sys.stderr); return ""
    output_buffer = []
    for r in range(new_height):
        line = [f"{bgr_to_rgb_ansi(resized_frame[r, c])}{asciify_pixel(resized_frame[r, c])}" for c in range(new_width)]
        output_buffer.append("".join(line) + "\033[0m")
    return "\n".join(output_buffer)

def render_frame_braille(frame, term_cols, term_rows, threshold):
    if term_cols <= 0 or term_rows <= 0: return ""
    braille_width = term_cols
    braille_height = term_rows - 1
    if braille_width <= 0 or braille_height <= 0: return ""
    target_width_pixels = braille_width * 2
    target_height_pixels = braille_height * 4
    try: resized_frame = cv2.resize(frame, (target_width_pixels, target_height_pixels), interpolation=cv2.INTER_AREA)
    except cv2.error as e: print(f"\nError resizing frame (Braille): {e}", file=sys.stderr); return ""
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    output_buffer = []
    for y_braille in range(braille_height):
        line = []
        for x_braille in range(braille_width):
            start_y, start_x = y_braille * 4, x_braille * 2
            color_block = resized_frame[start_y:start_y+4, start_x:start_x+2]
            avg_color_bgr = np.mean(color_block, axis=(0, 1))
            braille_dots = 0
            for (dy, dx), bit_value in DOT_MAP.items():
                pixel_y, pixel_x = start_y + dy, start_x + dx
                if 0 <= pixel_y < target_height_pixels and 0 <= pixel_x < target_width_pixels:
                    if gray_frame[pixel_y, pixel_x] > threshold: braille_dots |= bit_value
            braille_char = chr(BRAILLE_BASE_CODEPOINT + braille_dots)
            ansi_color = bgr_to_rgb_ansi(avg_color_bgr)
            line.append(f"{ansi_color}{braille_char}")
        output_buffer.append("".join(line))
    return "\n".join(output_buffer) + "\033[0m"

# --- Main Video Processing Function ---
def process_video(video_source, mode, threshold, audio_thread=None):
    is_webcam = isinstance(video_source, int)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        if is_webcam: print(f"Error: Could not open webcam (index: {video_source}).", file=sys.stderr)
        else: print(f"Error: Could not open video source: {video_source}", file=sys.stderr)
        if audio_thread and audio_thread.is_alive(): stop_audio_flag.set()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        if is_webcam: print("Warning: Webcam FPS unknown, defaulting to 30."); fps = 30
        else: print("Warning: Video FPS unknown, defaulting to 24."); fps = 24
    frame_delay = 1.0 / fps
    print(f"Target video FPS: {fps:.2f} (Delay: {frame_delay:.4f}s)")

    sys.stdout.write("\033[?25l\033[2J"); sys.stdout.flush() # Hide cursor, clear

    keep_running = True
    def signal_handler(sig, frame):
        nonlocal keep_running
        print("\nCtrl+C detected. Stopping...")
        keep_running = False
        stop_audio_flag.set()
    signal.signal(signal.SIGINT, signal_handler)

    last_term_size = (0, 0); term_cols, term_rows = 0, 0; frame_count = 0

    while keep_running:
        frame_start_time = time.monotonic()

        current_term_size = get_terminal_dimen()
        if current_term_size != last_term_size:
            last_term_size = current_term_size
            term_cols, term_rows = current_term_size
            sys.stdout.write("\033[2J"); sys.stdout.flush()
            if term_cols <= 0 or term_rows <= 0:
                 print("Terminal too small...");
                 while keep_running and (term_cols <= 0 or term_rows <= 0):
                     time.sleep(0.2); current_term_size = get_terminal_dimen()
                     term_cols, term_rows = current_term_size; last_term_size = current_term_size
                 if keep_running: sys.stdout.write("\033[2J"); sys.stdout.flush()
                 continue

        ret, frame = cap.read()
        if not ret:
            if is_webcam: print("Error reading webcam frame.", file=sys.stderr); time.sleep(0.5); continue
            else: print("End of video stream."); break

        frame_count += 1
        if is_webcam: frame = cv2.flip(frame, 1)

        frame_output = ""
        try:
            if mode == 'braille': frame_output = render_frame_braille(frame, term_cols, term_rows, threshold)
            elif mode == 'ascii': frame_output = render_frame_ascii(frame, term_cols, term_rows)
            else: print(f"Error: Unknown mode '{mode}'", file=sys.stderr); keep_running = False; continue
        except Exception as e: print(f"\nError rendering frame {frame_count}: {e}", file=sys.stderr); time.sleep(0.1); continue

        if frame_output: sys.stdout.write("\033[H"); sys.stdout.write(frame_output); sys.stdout.flush()

        frame_end_time = time.monotonic()
        elapsed_time = frame_end_time - frame_start_time
        sleep_time = frame_delay - elapsed_time
        if sleep_time > 0: time.sleep(sleep_time)

    # Cleanup
    print("Video loop finished. Releasing resources...")
    cap.release()
    stop_audio_flag.set()
    if audio_thread and audio_thread.is_alive():
        print("Waiting for audio thread to finish...")
        audio_thread.join(timeout=2.0)
        if audio_thread.is_alive():
            print("Audio thread did not stop gracefully, attempting force stop.")
            try: sd.stop()
            except Exception as e: print(f"Error force stopping audio: {e}", file=sys.stderr)

    sys.stdout.write("\033[?25h\033[0m\033[2J\033[H"); sys.stdout.flush() # Show cursor, reset, clear
    if is_webcam: print("Webcam stream stopped.")
    else: print("Playback finished.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play video (YouTube, local, webcam) in terminal with optional 8-bit audio.",
        formatter_class=argparse.RawTextHelpFormatter
        )
    parser.add_argument("video_source", help=f"YouTube URL, local file path, or '{WEBCAM_IDENTIFIER}'.")
    parser.add_argument("-m", "--mode", choices=['braille', 'ascii'], default='ascii', help="Rendering mode (default: ascii)")
    parser.add_argument("-t", "--threshold", type=int, default=DEFAULT_THRESHOLD, help=f"Braille brightness threshold (0-255, default: {DEFAULT_THRESHOLD})")
    # --- MODIFIED: Changed --no-audio to --audio ---
    parser.add_argument("--audio", action="store_true", # Defaults to False
                        help="Enable audio playback attempt (requires ffmpeg, pydub, sounddevice).")
    parser.add_argument("--no-8bit", action="store_true",
                        help="Disable 8-bit audio effect (play original audio if --audio is enabled).")

    args = parser.parse_args()

    if args.mode == 'braille' and (args.threshold < 0 or args.threshold > 255):
        print("Error: Threshold must be between 0 and 255.", file=sys.stderr); sys.exit(1)

    source_arg = args.video_source
    video_input = None
    audio_segment_obj = None
    processed_audio_segment = None
    final_sample_rate = 0
    audio_thread_handle = None
    downloaded_audio_path = None # Path if downloaded

    # Check if audio *can* be played based on imports
    can_play_audio = AUDIO_ENABLED and AudioSegment and sd
    # --- MODIFIED: Check if audio *should* be played based on --audio flag ---
    # It requires the --audio flag AND the necessary libraries must be present
    attempt_audio = args.audio and can_play_audio

    # --- Add a message if audio flag used but libraries missing ---
    if args.audio and not can_play_audio:
        print("Warning: --audio flag specified, but required audio libraries (pydub, sounddevice) or ffmpeg are missing/non-functional. Audio disabled.", file=sys.stderr)


    if source_arg.lower() == WEBCAM_IDENTIFIER:
        print("Webcam mode selected. Audio playback disabled.")
        video_input = 0
        attempt_audio = False # Ensure audio is off for webcam regardless of flag
    elif source_arg.startswith(('http://', 'https://')):
        # Call the function to get video URL and potentially download audio
        # Pass attempt_audio flag to control if download should happen
        video_input, downloaded_audio_path = get_video_url_and_download_audio(source_arg, attempt_audio) # Only attempts download if attempt_audio is True
        if not video_input:
            print("Failed to get video stream URL. Exiting.", file=sys.stderr); sys.exit(1)
        # If download failed or wasn't attempted, downloaded_audio_path will be None
        if attempt_audio and not downloaded_audio_path:
             print("Audio download failed or was skipped by function.")
             attempt_audio = False # Ensure audio logic is skipped later


    elif os.path.exists(source_arg): # Local file
        print(f"Processing local file: {source_arg}")
        video_input = source_arg
        # For local files, we load directly if audio is requested
        downloaded_audio_path = source_arg # Use the source path for loading attempt
        if not attempt_audio:
            print("Audio loading skipped (audio not enabled via --audio or libraries missing).")
    else:
        print(f"Error: Source not found or invalid: {source_arg}", file=sys.stderr); sys.exit(1)

    # --- Load Audio From File ---
    if attempt_audio and downloaded_audio_path:
        print(f"Loading audio from: {downloaded_audio_path}")
        try:
            audio_segment_obj = AudioSegment.from_file(downloaded_audio_path)
            print("Audio loaded successfully.")
        except CouldntDecodeError as e: print(f"Error decoding audio file: {e}", file=sys.stderr); attempt_audio = False
        except Exception as e: print(f"Unexpected error loading audio file: {e}", file=sys.stderr); attempt_audio = False
        finally:
            # Clean up temp file ONLY if it was downloaded (i.e., source was URL)
            if source_arg.startswith(('http://', 'https://')):
                 if downloaded_audio_path and os.path.exists(downloaded_audio_path):
                     # Check if it's the same as video_input (shouldn't be if downloaded)
                     if downloaded_audio_path != video_input:
                         print(f"Cleaning up temporary audio file: {downloaded_audio_path}")
                         try: os.remove(downloaded_audio_path)
                         except OSError as e_rem: print(f"Warning: Could not remove temp audio file: {e_rem}", file=sys.stderr)


    # --- Process and Start Audio Thread ---
    if attempt_audio and audio_segment_obj:
        if not args.no_8bit:
            processed_audio_segment, final_sample_rate = apply_8bit_effect(audio_segment_obj)
        else:
            print("Skipping 8-bit effect."); processed_audio_segment = audio_segment_obj; final_sample_rate = audio_segment_obj.frame_rate

        print(f"Reducing volume by -20 dB...")
        processed_audio_segment = processed_audio_segment - 20.0

        stop_audio_flag.clear()
        audio_thread_handle = threading.Thread(target=play_audio_thread, args=(processed_audio_segment, final_sample_rate), daemon=True)
        audio_thread_handle.start()
        time.sleep(0.5)
    elif attempt_audio and not audio_segment_obj: print("Audio not loaded, cannot play audio.")
    elif not attempt_audio: print("Audio playback is disabled or was not attempted.")

    # --- Start Video Processing ---
    if video_input is not None:
        process_video(video_input, args.mode, args.threshold, audio_thread_handle)
    else:
        print("Error: No valid video input.", file=sys.stderr)
        if audio_thread_handle and audio_thread_handle.is_alive(): stop_audio_flag.set(); audio_thread_handle.join(timeout=1.0)
        sys.exit(1)