from yt_dlp import YoutubeDL
from pathlib import Path

def time_to_seconds(time_str: str) -> int:
    """Convert time string (HH:MM:SS or MM:SS) to seconds"""
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

# def download_clip_(url: str, start_time: str, end_time: str, output_filename: str) -> None:
#     """
#     Download a specific portion of a YouTube video.

#     Args:
#         url (str): YouTube video URL
#         start_time (str): Start time in format HH:MM:SS
#         end_time (str): End time in format HH:MM:SS
#         output_filename (str): Name for the output file
#     """
#     try:
#         # Convert time strings to seconds
#         start_seconds: int = time_to_seconds(start_time)
#         end_seconds: int = time_to_seconds(end_time)
#         ydl_opts: dict = {
#             'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
#             'outtmpl': output_filename,
#             'external_downloader': 'ffmpeg',
#             'external_downloader_args': {
#                 'ffmpeg_i': ['-ss', str(start_seconds), '-t', str(end_seconds - start_seconds)]
#             }
#         }
#         with YoutubeDL(ydl_opts) as ydl:
#             ydl.download([url])
#         print(f"Successfully downloaded clip to {output_filename}")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         sys.exit(1)


def download_clip(url: str, start_time: str, end_time: str, output_filename: str):
    start_seconds: int = time_to_seconds(start_time)
    duration: int = time_to_seconds(end_time) - start_seconds
    base_path = Path.home() / "Downloads" / "parts"
    base_path.mkdir(parents=True, exist_ok=True)
    ydl_opts: dict = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': str(base_path / output_filename),
        'external_downloader': 'ffmpeg',
        'external_downloader_args': [
            '-ss', str(start_seconds),
            '-t', str(duration)
        ]
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Successfully downloaded clip to {output_filename}")


def merge_videos():
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    from pathlib import Path
    home = str(Path.home())
    # Load video clips
    clip1 = VideoFileClip(f"{home}/clip.mp4")
    clip2 = VideoFileClip(f"{home}/clip2.mp4")

    # Reduce volume of first clip by half
    clip1 = clip1.volumex(0.5)
    clip2 = clip2.volumex(2.0)

    # Concatenate clips
    final_clip = concatenate_videoclips([clip1, clip2])
    # Write output
    final_clip.write_videofile(f"{home}/clip_merged.mp4")
    # Close clips to free up resources
    clip1.close()
    clip2.close()
    final_clip.close()


if __name__ == "__main__":
    # Example usage
    # url: str = "https://www.youtube.com/watch?v"
    # start_time: str = "1:25:51"
    # end_time: str = "1:28:17"
    # output_filename: str = "clip5.mp4"
    # download_clip(url, start_time, end_time, output_filename)
    pass
