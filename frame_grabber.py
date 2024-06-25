import time
import cv2
import redis
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RTSPS stream configuration
BASE_URL = "rtsps://192.168.0.1:7441/"
CAMERA_IDS = [
    "iY9STaEt7K9vS8yJ",
    # Add the rest of your camera identifiers here
]
TOTAL_CAMERAS = len(CAMERA_IDS)
TARGET_FPS = 1 / 60  # 1 frame per minute
MAX_WORKERS = 44  # Utilizing all 44 CPU cores

# Redis configuration
REDIS_HOST = '192.168.0.71'
REDIS_PORT = 6379
REDIS_QUEUE = 'frame_queue'

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

def grab_and_queue_frame(camera_id, camera_index):
    """Grab a single frame from the given camera URL and push it to the Redis queue."""
    camera_url = f"{BASE_URL}{camera_id}?enableSrtp"
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        logging.error(f"Failed to open stream for camera {camera_index}")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logging.error(f"Failed to grab frame from camera {camera_index}")
        return
    
    # Encode frame as PNG for better quality
    _, buffer = cv2.imencode('.png', frame)
    png_as_text = buffer.tobytes()
    
    # Create a dictionary with metadata
    frame_data = {
        'camera_id': camera_id,
        'camera_index': camera_index,
        'timestamp': datetime.now().isoformat(),
        'frame': png_as_text
    }
    
    # Push to Redis queue
    redis_client.rpush(REDIS_QUEUE, str(frame_data))
    logging.info(f"Frame from camera {camera_index} pushed to Redis queue")

def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while True:
            start_time = time.time()
            
            # Submit tasks for all cameras
            futures = [executor.submit(grab_and_queue_frame, camera_id, i+1) 
                       for i, camera_id in enumerate(CAMERA_IDS)]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
            
            # Calculate sleep time to maintain target FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1 / TARGET_FPS) - elapsed_time)
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()