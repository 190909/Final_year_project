import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



train_data_dir = 'C:/Users/91830/OneDrive/Desktop/Project_dataset/Train'
val_data_dir = 'C:/Users/91830/OneDrive/Desktop/Project_dataset/Validation'



img_size = 224
batch_size = 32


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_data_dir,
                                                target_size=(img_size, img_size),
                                                batch_size=batch_size,
                                                class_mode='categorical')


model=tf.keras.models.load_model('C:/Users/91830/OneDrive/Desktop/Project_dataset/Models/Densenet169.h5')

import cv2
import os

video_path = 'C:/Users/91830/OneDrive/Desktop/constant.mp4'
output_dir = 'C:/Users/91830/OneDrive/Desktop/frames'

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    frame_name = f"frame_{i}.jpg"
    frame_path = os.path.join(output_dir, frame_name)
    cv2.imwrite(frame_path, frame)


import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input

for i in range(frame_count):
    frame_name = f"frame_{i}.jpg"
    frame_path = os.path.join(output_dir, frame_name)
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)


    prediction = model.predict(img)
if prediction[0][0] > prediction[0][1]:
    result="Fire"
    print('Fire Detected')
else:
    result="No fire"
    print('No Fire Detected')

    



import cv2
import numpy as np
import matplotlib.pyplot as plt

if result=="Fire":
    cap = cv2.VideoCapture('C:/Users/91830/OneDrive/Desktop/Fire_1.mp4')

    interval = 10

    intensity_values = []




    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('fire_intensity.mp4', fourcc, 20.0, (640, 480))

    # Get the dimensions of the frame
    height, width, channels = frame.shape
    # Calculate the total number of pixels in the frame
    num_pixels = height * width

    print("Total number of pixels in the frames of the video:", num_pixels)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if cap.get(1) % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            fire_pixels = np.count_nonzero(thresh)
            intensity_values.append(fire_pixels)
            
            print("Number of pixels containing fire       :", fire_pixels)
            intensity=fire_pixels/num_pixels
            print("The intensity of fire in the frame:              ",intensity)
            
            
            out.write(cv2.putText(frame, f'Fire Intensity: {intensity:.2f}', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA))
            # Display the frame
            cv2.imshow('frame', frame)
        
    overall_frames=sum(intensity_values)/len(intensity_values)
    overall_intensity= overall_frames/num_pixels


    print("The overall intensity throughout the video is:    " ,overall_intensity)


    import cv2
import numpy as np

# load the video
cap = cv2.VideoCapture('C:/Users/91830/OneDrive/Desktop/constant.mp4')

# initialize variables
area_values = []
frame_count = 0

# loop over the frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # increment frame count
    frame_count += 1
    
    # process every 10 frames
    if frame_count % 10 == 0:
        # preprocess the frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # detect the fire region
        fire_region = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
        fire_region = cv2.erode(fire_region, None, iterations=2)
        fire_region = cv2.dilate(fire_region, None, iterations=4)
        contours, hierarchy = cv2.findContours(fire_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # calculate the area of the fire
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            area_values.append(area)
        else:
            area_values.append(0)

# calculate the percentage change in the area of fire over the 10-frame interval
area_diff = np.diff(area_values)
percent_change = (area_diff / area_values[:-1]) * 100

 # determine if the fire is spreading, decreasing or no fire
if percent_change[-1] > 50:
    status="spread"
    print("The fire is spreading.")
elif percent_change[-1] < -50:
    print("The fire is decreasing.")
    status="The fire is decreasing"
elif abs(percent_change[-1]) < 10:
        print("The fire is constant.")
        status="The fire is constant"
else:
    print("No fire detected.")

# calculate the rate of change in the area of fire over the 10-frame interval
rate_of_change = area_diff / np.mean(area_values[:-1])
print(f"The rate of change in the area of fire is {rate_of_change} per 10 frames.")




from twilio.rest import Client
if status == "spread":
# Your Account SID and Auth Token from twilio.com/console
    account_sid = 'AC613c9e2d1f380999dd470fe5a50620f6'
    auth_token = 'c0342910796e9b3e4e28ddca694da106'
    client = Client(account_sid, auth_token)

    message = client.messages \
                    .create(
                        body="Alert! Fire detected in your forest area and its starting to spread..!! Take necessary actions!!!!!!!",
                        from_='+15855977296',
                        to='+918300661982'
                    )

    print(message.sid)

   
        


