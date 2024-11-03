from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path

def merge_videos():
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
    merge_videos()
