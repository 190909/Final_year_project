# import cv2
# import os

# # Path to the folder containing videos
# video_folder = "C:/Users/91830/OneDrive/Desktop/Project_dataset/Fire_dataset"

# # Path to the folder where frames will be saved
# frame_folder = "C:/Users/91830/OneDrive/Desktop/Project_dataset/Fire_dataset/frames"

# # Loop through all the files in the video folder
# for filename in os.listdir(video_folder):
#     # Check if the file is a video file
#     if filename.endswith(".mp4") or filename.endswith(".avi"):
#         # Open the video file
#         video_path = os.path.join(video_folder, filename)
#         cap = cv2.VideoCapture(video_path)

#         # Create a folder for the frames
#         frame_path = os.path.join(frame_folder, os.path.splitext(filename)[0])
#         if not os.path.exists(frame_path):
#             os.makedirs(frame_path)

#         # Loop through the frames and save them as images
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_count += 1
#             frame_filename = f"frame{frame_count}.jpg"
#             frame_pathname = os.path.join(frame_path, frame_filename)
#             cv2.imwrite(frame_pathname, frame)

#         # Release the video file
#         cap.release()

import os
import cv2

# Path to folder containing videos
video_folder = 'C:/Users/91830/OneDrive/Desktop/Project_dataset/Non_Fire_dataset'

# Path to folder to store frames
frame_folder = 'C:/Users/91830/OneDrive/Desktop/Project_dataset/Non_Fire_dataset/Frames'

# Frame rate for output frames
frame_rate = 7

# Loop through all files in video folder
for filename in os.listdir(video_folder):
    if filename.endswith('.mp4') or filename.endswith('.webm'):
        # Get full path to video file
        video_path = os.path.join(video_folder, filename)
        
        # Create folder to store frames for this video
        video_frame_folder = os.path.join(frame_folder, filename[:-4])
        os.makedirs(video_frame_folder, exist_ok=True)
        
        # Open video file with OpenCV
        cap = cv2.VideoCapture(video_path)
        
        # Get frame rate and calculate frame interval
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = round(fps / frame_rate)
        
        # Loop through all frames in video
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Check if end of video or error reading frame
            if not ret:
                break
            
            # Only process every interval-th frame
            if count % interval == 0:
                # Construct path for output frame
                frame_path = os.path.join(video_frame_folder, f'frame_{count}.jpg')
                
                # Save frame to file
                cv2.imwrite(frame_path, frame)
            
            count += 1
        
        # Release OpenCV video capture object
        cap.release()
