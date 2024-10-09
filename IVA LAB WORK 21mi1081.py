#!/usr/bin/env python
# coding: utf-8

# In[1]:



#IMAGEAND VIDEOANALYTICS 
#LAB TASK 
#NAME: DOPPALAPUDI RESHMA
#REGN : 21MIA1081
#Course: CSE4076


#Load Video:
#Load the provided video file.
import cv2

# Open the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Display the frame
            cv2.imshow('Frame', frame)
            
            # Press 'q' to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()


# In[1]:


#Frame Extraction:
import cv2
import os

# Path to the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"

# Directory where frames will be saved
output_dir = "extracted_frames"

# Create directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

frame_count = 0

# Read until the video is completed
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Every frame will be saved with its frame number
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        
        # Save the frame as a .jpg image
        cv2.imwrite(frame_filename, frame)
        
        print(f"Saved {frame_filename}")
        
        frame_count += 1
        
        # To limit extraction (optional, e.g., extract every 30 frames)
        # if frame_count % 30 == 0:
        #     cv2.imwrite(frame_filename, frame)

    else:
        break

# Release the video capture object
cap.release()
print(f"Extracted {frame_count} frames and saved in {output_dir}.")


# In[2]:


#Spatio-Temporal Segmentation:
#Perform segmentation on each frame using a technique like color thresholding or edge detection.
import cv2
import os
import numpy as np

# Path to the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"

# Directories for saving extracted and segmented frames
output_dir_extracted = "extracted_frames"
output_dir_segmented = "segmented_frames"

# Create directories if they don't exist
if not os.path.exists(output_dir_extracted):
    os.makedirs(output_dir_extracted)
if not os.path.exists(output_dir_segmented):
    os.makedirs(output_dir_segmented)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Define the color range for segmentation (HSV format)
lower_color = np.array([40, 40, 40])   # Lower bound (example: green)
upper_color = np.array([70, 255, 255]) # Upper bound (example: green)

frame_count = 0

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        # Save the extracted frame
        frame_filename = os.path.join(output_dir_extracted, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved extracted frame: {frame_filename}")
        
        # Convert the frame to HSV for color thresholding
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply color thresholding
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to grayscale for edge detection
        gray_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_frame, 100, 200)
        
        # Save the segmented frame (edges and color segmentation)
        segmented_filename = os.path.join(output_dir_segmented, f"segmented_{frame_count:04d}.jpg")
        cv2.imwrite(segmented_filename, edges)
        print(f"Saved segmented frame with edge detection: {segmented_filename}")
        
        frame_count += 1
    else:
        break

# Release the video capture object
cap.release()
print(f"Processed {frame_count} frames. Extracted frames saved in '{output_dir_extracted}', segmented frames saved in '{output_dir_segmented}'.")


# In[3]:


#Track the segmented objects across frames to observe changes in motion and shape.
import cv2
import os
import numpy as np

# Path to the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"

# Directories for saving segmented frames
output_dir_segmented = "tracked_frames"

# Create directory for saving tracked frames
if not os.path.exists(output_dir_segmented):
    os.makedirs(output_dir_segmented)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Define the color range for segmentation (HSV format)
lower_color = np.array([40, 40, 40])   # Example for green
upper_color = np.array([70, 255, 255]) # Example for green

frame_count = 0

# To store object centroids across frames for tracking
previous_centroids = []

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Convert the frame to HSV for color thresholding
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply color thresholding
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to grayscale for contour detection
        gray_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply a threshold to get binary image (for contour detection)
        _, thresh = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours of the segmented objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_centroids = []

        # Loop over each contour
        for contour in contours:
            # Ignore small areas to filter out noise
            if cv2.contourArea(contour) > 500:
                # Get bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw bounding box around the object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate centroid of the object
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    current_centroids.append((cx, cy))

                    # Draw the centroid
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Track changes in the bounding box (size)
                bounding_box_area = w * h
                cv2.putText(frame, f"Area: {bounding_box_area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Track centroid motion across frames
        if previous_centroids:
            for prev_cx, prev_cy in previous_centroids:
                # Draw lines to track movement from previous to current centroid
                for curr_cx, curr_cy in current_centroids:
                    cv2.line(frame, (prev_cx, prev_cy), (curr_cx, curr_cy), (255, 0, 0), 2)

        # Update previous centroids for the next frame
        previous_centroids = current_centroids

        # Save the frame with bounding boxes and motion tracking
        output_frame_path = os.path.join(output_dir_segmented, f"tracked_{frame_count:04d}.jpg")
        cv2.imwrite(output_frame_path, frame)

        print(f"Processed frame {frame_count}: tracked objects and saved to {output_frame_path}")
        
        frame_count += 1
    else:
        break

# Release the video capture object
cap.release()
print(f"Tracking completed. Processed {frame_count} frames.")


# In[4]:


#Identify the regions that remain consistent over time (foreground vs. background segmentation).
import cv2
import os

# Path to the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"

# Directories for saving foreground and background frames
output_dir_foreground = "foreground_frames"
output_dir_background = "background_frames"

# Create directories if they don't exist
if not os.path.exists(output_dir_foreground):
    os.makedirs(output_dir_foreground)
if not os.path.exists(output_dir_background):
    os.makedirs(output_dir_background)

# Create a background subtractor object
back_sub = cv2.createBackgroundSubtractorMOG2()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Apply the background subtractor to get the foreground mask
        fg_mask = back_sub.apply(frame)

        # Apply thresholding to the mask to binarize it
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Optionally apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Save the foreground mask frame
        fg_mask_filename = os.path.join(output_dir_foreground, f"foreground_{frame_count:04d}.jpg")
        cv2.imwrite(fg_mask_filename, fg_mask)

        # Save the original frame with mask overlay for visualization
        colored_foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)
        background_filename = os.path.join(output_dir_background, f"background_{frame_count:04d}.jpg")
        cv2.imwrite(background_filename, colored_foreground)

        print(f"Processed frame {frame_count}: saved foreground mask and background overlay.")
        
        frame_count += 1
    else:
        break

# Release the video capture object
cap.release()
print(f"Foreground-background segmentation completed. Processed {frame_count} frames.")


# In[5]:


#Scene Cut Detection:
#Use pixel-based comparison or histogram differences between consecutive frames to detect abrupt changes (hard cuts)
import cv2
import os
import numpy as np

# Path to the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"

# Directory to save frames with detected cuts
output_dir_cuts = "scene_cuts"
if not os.path.exists(output_dir_cuts):
    os.makedirs(output_dir_cuts)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, prev_frame = cap.read()

# Convert the first frame to HSV color space and calculate its histogram
prev_frame_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
prev_hist = cv2.calcHist([prev_frame_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
cv2.normalize(prev_hist, prev_hist)

frame_count = 1
cut_frames = []

# Process each frame
while True:
    ret, curr_frame = cap.read()

    if not ret:
        break

    # Convert current frame to HSV and calculate its histogram
    curr_frame_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    curr_hist = cv2.calcHist([curr_frame_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(curr_hist, curr_hist)

    # Compare histograms using correlation
    correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

    # If the correlation is below a certain threshold, it indicates a scene cut
    threshold = 0.5  # Adjust this value based on your needs
    if correlation < threshold:
        print(f"Scene cut detected between frames {frame_count - 1} and {frame_count}")
        cut_frames.append(frame_count)
        
        # Save the frame with detected cut
        cut_frame_path = os.path.join(output_dir_cuts, f"cut_{frame_count:04d}.jpg")
        cv2.imwrite(cut_frame_path, curr_frame)

    # Update previous frame and histogram for the next iteration
    prev_frame_hsv = curr_frame_hsv
    prev_hist = curr_hist

    frame_count += 1

# Release the video capture object
cap.release()
print(f"Scene cut detection completed. Detected cuts at frames: {cut_frames}.")


# In[6]:


#Detect gradual scene transitions (Soft cuts) by analyzing frame-to-frame intensity changes over time.
import cv2
import os
import numpy as np

# Path to the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"

# Directory to save frames with detected transitions
output_dir_transitions = "soft_cuts"
if not os.path.exists(output_dir_transitions):
    os.makedirs(output_dir_transitions)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
previous_frame_intensity = None
soft_cut_frames = []

# Process each frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the average intensity of the frame
    avg_intensity = np.mean(gray_frame)

    # Detect gradual scene transitions by comparing with the previous frame's intensity
    if previous_frame_intensity is not None:
        intensity_diff = abs(avg_intensity - previous_frame_intensity)
        
        # Define a threshold for detecting soft cuts
        threshold = 10  # Adjust this value based on the content and frame rate
        if intensity_diff < threshold:
            print(f"Soft cut detected between frames {frame_count - 1} and {frame_count}")
            soft_cut_frames.append(frame_count)
            
            # Save the frame with detected transition
            transition_frame_path = os.path.join(output_dir_transitions, f"soft_cut_{frame_count:04d}.jpg")
            cv2.imwrite(transition_frame_path, frame)

    # Update previous frame intensity for the next iteration
    previous_frame_intensity = avg_intensity
    frame_count += 1

# Release the video capture object
cap.release()
print(f"Soft cut detection completed. Detected transitions at frames: {soft_cut_frames}.")


# In[9]:


#Mark Scene Cuts:
#Highlight the frames where scene cuts are detected.
#Create a summary displaying the detected scene boundaries
import cv2
import os
import numpy as np

# Path to the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"

# Directory to save frames with detected cuts
output_dir_cuts = "marked_scene_cuts"
if not os.path.exists(output_dir_cuts):
    os.makedirs(output_dir_cuts)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Failed to read video.")
    cap.release()
    exit()

# Convert the first frame to HSV color space and calculate its histogram
prev_frame_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
prev_hist = cv2.calcHist([prev_frame_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
cv2.normalize(prev_hist, prev_hist)

frame_count = 1
cut_frames = []

# Process each frame
while True:
    ret, curr_frame = cap.read()

    if not ret:
        break

    # Convert current frame to HSV and calculate its histogram
    curr_frame_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    curr_hist = cv2.calcHist([curr_frame_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(curr_hist, curr_hist)

    # Compare histograms using correlation
    correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

    # If the correlation is below a certain threshold, it indicates a scene cut
    threshold = 0.5  # Adjust this value based on your needs
    if correlation < threshold:
        print(f"Scene cut detected between frames {frame_count - 1} and {frame_count}")
        cut_frames.append(frame_count)

        # Draw a rectangle or overlay text on the current frame to mark the cut
        cv2.rectangle(curr_frame, (10, 10), (200, 50), (0, 0, 255), -1)  # Rectangle on the top left
        cv2.putText(curr_frame, "Scene Cut Detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Update previous frame and histogram for the next iteration
    prev_frame_hsv = curr_frame_hsv
    prev_hist = curr_hist

    # Save the marked frame
    marked_frame_path = os.path.join(output_dir_cuts, f"marked_{frame_count:04d}.jpg")
    cv2.imwrite(marked_frame_path, curr_frame)

    frame_count += 1

# Release the video capture object
cap.release()

# Create a summary report of detected scene cuts
summary_file_path = "scene_cut_summary.txt"
with open(summary_file_path, 'w') as summary_file:
    summary_file.write("Detected Scene Cuts:\n")
    summary_file.write("====================\n")
    for frame in cut_frames:
        summary_file.write(f"Scene cut at frame: {frame}\n")

print(f"Marked scene cuts detection completed. Detected cuts at frames: {cut_frames}.")
print(f"Summary report saved to: {summary_file_path}.")


# In[11]:


#Result Visualization:
#Display frames where scene cuts are identified and show segmentation results for selected frames.

import cv2
import os
import numpy as np

# Path to the video file
video_path = r"C:\Users\user\Downloads\1507868-uhd_3840_2160_25fps.mp4"

# Directory to save frames with detected cuts
output_dir_cuts = "marked_scene_cuts"
if not os.path.exists(output_dir_cuts):
    os.makedirs(output_dir_cuts)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Read the first frame and initialize variables
ret, prev_frame = cap.read()
if not ret:
    print("Failed to read video.")
    cap.release()
    exit()

prev_frame_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
prev_hist = cv2.calcHist([prev_frame_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
cv2.normalize(prev_hist, prev_hist)

frame_count = 1
cut_frames = []

# Process each frame
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    # Convert current frame to HSV and calculate its histogram
    curr_frame_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    curr_hist = cv2.calcHist([curr_frame_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(curr_hist, curr_hist)

    # Compare histograms using correlation
    correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

    # If the correlation is below a certain threshold, it indicates a scene cut
    threshold = 0.5  # Adjust this value based on your needs
    if correlation < threshold:
        print(f"Scene cut detected at frame {frame_count}")
        cut_frames.append(frame_count)

        # Draw a rectangle or overlay text on the current frame to mark the cut
        cv2.rectangle(curr_frame, (10, 10), (200, 50), (0, 0, 255), -1)  # Rectangle on the top left
        cv2.putText(curr_frame, "Scene Cut Detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Perform segmentation on the marked frame (example: color thresholding)
        # Here, we'll apply a simple color threshold to extract the red color
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(curr_frame_hsv, lower_red, upper_red)

        # Bitwise-AND mask and original image to get the segmented result
        segmented_result = cv2.bitwise_and(curr_frame, curr_frame, mask=mask)

        # Display the marked frame and segmented result
        cv2.imshow(f"Marked Frame {frame_count}", curr_frame)
        cv2.imshow(f"Segmented Result for Frame {frame_count}", segmented_result)

        # Save marked frame to disk
        marked_frame_path = os.path.join(output_dir_cuts, f"marked_{frame_count:04d}.jpg")
        cv2.imwrite(marked_frame_path, curr_frame)

    # Update previous frame and histogram for the next iteration
    prev_frame_hsv = curr_frame_hsv
    prev_hist = curr_hist
    frame_count += 1

# Release the video capture object
cap.release()

# Create a summary report of detected scene cuts
summary_file_path = "scene_cut_summary.txt"
with open(summary_file_path, 'w') as summary_file:
    summary_file.write("Detected Scene Cuts:\n")
    summary_file.write("====================\n")
    for frame in cut_frames:
        summary_file.write(f"Scene cut at frame: {frame}\n")

print(f"Marked scene cuts detection completed. Detected cuts at frames: {cut_frames}.")
print(f"Summary report saved to: {summary_file_path}.")

# Wait for a key press before closing the windows
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




