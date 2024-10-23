

from yt_dlp import YoutubeDL
import sys

def time_to_seconds(time_str):
    """Convert time string (HH:MM:SS or MM:SS) to seconds"""
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def download_clip(url, start_time, end_time, output_filename):
    """
    Download a specific portion of a YouTube video.
    
    Args:
        url (str): YouTube video URL
        start_time (str): Start time in format HH:MM:SS
        end_time (str): End time in format HH:MM:SS
        output_filename (str): Name for the output file
    """
    try:
        # Convert time strings to seconds
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': output_filename,
            'external_downloader': 'ffmpeg',
            'external_downloader_args': {
                'ffmpeg_i': ['-ss', str(start_seconds), '-t', str(end_seconds - start_seconds)]
            }
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Successfully downloaded clip to {output_filename}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Example usage
    url = "https://www.youtube.com/watch?dsdg"
    start_time = "1:19:00"
    end_time = "1:19:40"
    output_filename = "clip.mp4"    
    download_clip(url, start_time, end_time, output_filename)

