import time
import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RTSPS stream configuration
BASE_URL = "rtsps://192.168.0.1:7441/"
CAMERA_IDS = [
    "fBsMBKQxVTOw7WwR",
    # Add the rest of your camera identifiers here
]
TOTAL_CAMERAS = len(CAMERA_IDS)
TARGET_FPS = 1 / 60  # 1 frame per minute
MAX_WORKERS = 44  # Utilizing all 44 CPU cores

# Directory to save frames
SAVE_DIR = 'captured_frames'
os.makedirs(SAVE_DIR, exist_ok=True)

def grab_and_save_frame(camera_id, camera_index):
    """Grab a single frame from the given camera URL and save it locally."""
    camera_url = f"{BASE_URL}{camera_id}?enableSrtp"
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        logging.error(f"Failed to open stream for camera {camera_index}")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logging.error(f"Failed to grab frame from camera {camera_index}")
        return
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"camera_{camera_index}_{timestamp}.png"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Save frame as PNG
    cv2.imwrite(filepath, frame)
    logging.info(f"Frame from camera {camera_index} saved to {filepath}")
    
    # Save a copy as 'latest' for easy viewing
    latest_filepath = os.path.join(SAVE_DIR, f"camera_{camera_index}_latest.png")
    cv2.imwrite(latest_filepath, frame)

def display_latest_frames():
    """Display the latest frame from each camera."""
    latest_frames = [cv2.imread(os.path.join(SAVE_DIR, f"camera_{i}_latest.png")) 
                     for i in range(1, TOTAL_CAMERAS + 1)]
    valid_frames = [frame for frame in latest_frames if frame is not None]
    
    if not valid_frames:
        print("No frames available to display.")
        return
    
    # Create a grid of images
    rows = (len(valid_frames) + 3) // 4  # 4 images per row
    grid = []
    for i in range(0, len(valid_frames), 4):
        row = valid_frames[i:i+4]
        if len(row) < 4:
            row += [np.zeros_like(valid_frames[0])] * (4 - len(row))  # Pad with black images
        grid.append(np.hstack(row))
    
    grid_image = np.vstack(grid)
    
    # Resize if the image is too large
    scale = min(1.0, 1920 / grid_image.shape[1])  # Limit width to 1920 pixels
    if scale < 1.0:
        grid_image = cv2.resize(grid_image, None, fx=scale, fy=scale)
    
    cv2.imshow("Latest Frames", grid_image)
    cv2.waitKey(1)  # Update the window

def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while True:
            start_time = time.time()
            
            # Submit tasks for all cameras
            futures = [executor.submit(grab_and_save_frame, camera_id, i+1) 
                       for i, camera_id in enumerate(CAMERA_IDS)]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
            
            # Display the latest frames
            display_latest_frames()
            
            # Calculate sleep time to maintain target FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1 / TARGET_FPS) - elapsed_time)
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()