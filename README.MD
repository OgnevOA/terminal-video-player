Okay, here is the updated `README.md` reflecting that audio is now off by default and requires the `--audio` flag to be enabled.

# Terminal Video Player (ASCII/Braille)

This Python script allows you to play YouTube videos, local video files, or your webcam feed directly in your terminal, rendering them using either colored ASCII characters or colored Unicode Braille patterns. Optionally, it can attempt to play audio (with an 8-bit effect) alongside the video.

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

*   **Dual Rendering Modes:**
    *   **ASCII Mode:** Renders video using a range of ASCII characters based on pixel brightness. Uses full color for each character cell. (Default)
    *   **Braille Mode:** Renders video using Unicode Braille patterns (U+2800-U+28FF). Each character represents a 2x4 pixel block. Uses the average color of the block.
*   **Multiple Video Sources:**
    *   YouTube URLs
    *   Local video files
    *   Webcam feed
*   **Optional Audio Playback:**
    *   Audio is **disabled by default**.
    *   Use the `--audio` flag to enable audio fetching and playback.
    *   Applies a basic "8-bit" effect (downsampling, bit reduction) by default.
    *   Use `--no-8bit` flag (with `--audio`) to play original audio.
    *   *Note: Audio synchronization with terminal video is experimental and likely to drift.*
*   **Efficient Downloading:** Uses `yt-dlp` to fetch video/audio information and directly download audio streams (when enabled) for potentially better performance.
*   **TrueColor Support:** Utilizes ANSI 24-bit TrueColor escape codes for accurate color representation (requires a compatible terminal).
*   **Dynamic Resizing:** Adapts the output resolution to the current terminal size.
*   **Framerate Control:** Attempts to maintain the original video framerate.
*   **Cross-Platform:** Should work on Linux, macOS, and Windows (provided dependencies and terminal requirements are met).


## Requirements

1.  **Python 3.x:** The script is written for Python 3.
2.  **yt-dlp:** The `yt-dlp` command-line tool must be installed and accessible in your system's PATH.
    *   Installation: [https://github.com/yt-dlp/yt-dlp#installation](https://github.com/yt-dlp/yt-dlp#installation)
3.  **Python Libraries (Core):**
    *   `opencv-python`: For video processing.
    *   `numpy`: Required by OpenCV.
4.  **Python Libraries (Optional - for Audio):** Required only if using the `--audio` flag.
    *   `pydub`: For audio manipulation and effects.
    *   `sounddevice`: For audio playback.
    *   `requests`: For basic URL checking (though download uses yt-dlp).
5.  **External Dependency (Optional - for Audio):** Required only if using the `--audio` flag.
    *   **FFmpeg:** `pydub` relies *heavily* on FFmpeg being installed and accessible in your system's PATH for loading, converting, and applying effects to audio.
        *   Installation: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use a package manager (`sudo apt install ffmpeg`, `brew install ffmpeg`, etc.) and ensure the `bin` directory containing `ffmpeg.exe`/`ffmpeg` is added to your system PATH.

## Installation

1.  **Install `yt-dlp`:** Follow the official instructions linked above. Verify with `yt-dlp --version`.
2.  **Install FFmpeg (if using audio):** Follow instructions for your OS. Verify with `ffmpeg -version`. Add its `bin` directory to your PATH.
3.  **Install Core Python Libraries:**
    ```bash
    pip install opencv-python numpy
    ```
4.  **Install Optional Audio Python Libraries (if using audio):**
    ```bash
    pip install pydub sounddevice requests
    ```
    *(Note: Consult official documentation if you encounter issues with specific library installations.)*

5.  **Save the Script:** Save the Python code provided previously as a file, e.g., `yt_terminal_player.py`.

## Usage

Run the script from your terminal using the following format:

```bash
python yt_terminal_player.py [options] <video_source>
```

**Arguments:**

*   `<video_source>`: (Required) YouTube URL, path to a local video file, or the keyword `webcam`. **Enclose URLs in quotes if they contain special characters.**
*   `-m <mode>`, `--mode <mode>`: (Optional) Rendering mode: `ascii` (Default) or `braille`.
*   `-t <threshold>`, `--threshold <threshold>`: (Optional) Brightness threshold (0-255) for Braille mode dots. (Default: `120`)
*   `--audio`: (Optional) **Enable** audio playback attempt. Requires `ffmpeg`, `pydub`, `sounddevice`. (Default: Disabled)
*   `--no-8bit`: (Optional) If `--audio` is used, this flag disables the 8-bit effect and attempts to play the original audio.

**Examples:**

*   Play a YouTube video using ASCII mode (no audio - default):
    ```bash
    python yt_terminal_player.py "https://www.youtube.com/watch?v=..."
    ```
*   Play a local file using Braille mode (no audio - default):
    ```bash
    python yt_terminal_player.py -m braille my_video.mp4
    ```
*   Play a YouTube video using Braille mode **with 8-bit audio enabled**:
    ```bash
    python yt_terminal_player.py --audio -m braille "https://www.youtube.com/watch?v=..."
    ```
*   Play a local video file **with original audio enabled** (no 8-bit effect):
    ```bash
    python yt_terminal_player.py --audio --no-8bit my_cool_video.mp4
    ```
*   View webcam feed using ASCII mode (audio is always disabled for webcam):
    ```bash
    python yt_terminal_player.py webcam
    ```

**Stopping Playback:** Press `Ctrl+C` to stop the script.

## Terminal Setup Notes

*   **TrueColor Support:** Your terminal emulator **must** support 24-bit ANSI TrueColor for colors to display correctly.
*   **Braille Font Support:** For **Braille mode**, your terminal font must include the Unicode Braille Patterns block (U+2800-U+28FF). (e.g., DejaVu Sans Mono, Noto Sans Mono, Cascadia Code, Nerd Fonts).

## Limitations

*   **Audio Synchronization:** If audio is enabled (`--audio`), it starts playing when the video rendering begins. However, due to the variable speed of terminal rendering, the audio **will likely drift out of sync** with the video over time. This is an inherent limitation.
*   **Audio Quality:** The default 8-bit effect is basic and can sound harsh. Original audio quality depends on the source fetched by `yt-dlp`.
*   **Performance:** Rendering video in the terminal is CPU-intensive. Performance depends on CPU, terminal emulator, and terminal size.
*   **Aspect Ratio:** Does not add black bars to preserve the original video aspect ratio if it differs from the terminal's character grid shape.
*   **No Audio for Webcam:** The webcam mode does not support audio capture or playback.

## Contributing

Contributions welcome! Feel free to open issues or pull requests.

## License

This project is licensed under the MIT License.