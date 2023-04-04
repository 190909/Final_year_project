import os
from moviepy.video.io.VideoFileClip import VideoFileClip


# Path to folder containing videos
video_folder = 'C:/Users/91830/OneDrive/Desktop/Project_dataset/Non_Fire_dataset/Original'

# Maximum duration for output videos
max_duration = 15  # in seconds

# Loop through all files in video folder
for filename in os.listdir(video_folder):
    if filename.endswith('.mp4') or filename.endswith(".webm"):
        # Get full path to video file
        video_path = os.path.join(video_folder, filename)
        
        # Create output filename for cropped video
        cropped_filename = f'{filename[:-4]}_cropped.mp4'
        cropped_path = os.path.join(video_folder, cropped_filename)
        
        # Open video file with moviepy
        clip = VideoFileClip(video_path)
        
        # Get duration of video
        duration = clip.duration
        
        # Crop video if longer than maximum duration
        if duration > max_duration:
            # Set end time to maximum duration
            end_time = max_duration
            
            # Crop video from 0 to end_time
            cropped_clip = clip.subclip(0, end_time)
        else:
            # Don't crop video if already shorter than maximum duration
            cropped_clip = clip
        
        # Write cropped video to file
        cropped_clip.write_videofile(cropped_path)
        
        # Close moviepy clip object
        clip.close()
