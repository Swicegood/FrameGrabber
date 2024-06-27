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
    "I6Dvhhu1azyV9rCu",
    "PxnDZaXu2awYbMmS",
    "oaQllpjP0sk94nCV",
    "mKlJgNx7tXwalch1",
    "rHWz9GRDFxrOZF7b",
    "LRqgKMMjjJbNEeyE",
    "94uZsJ2yIouIXp2x",
    "5SJZivf8PPsLWw2n",
    "g8rHNVCflWO1ptKN",
    "t3ZIWTl9jZU1JGEI",
    "iY9STaEt7K9vS8yJ",
    "jlNNdFFvhQ2o2kmn",
    "IOKAu7MMacLh79zn",
    "sHlS7ewuGDEd2ef4",
    "OSF13XTCKhpIkyXc",
    "jLUEC60zHGo7BXfj"
]
TOTAL_CAMERAS = len(CAMERA_IDS)
TARGET_FPS = 1 / 1020  # 1 frame per minute
MAX_WORKERS = 44  # Utilizing all 44 CPU cores

# Redis configuration
REDIS_HOST = '192.168.0.71'
REDIS_PORT = 6379
REDIS_QUEUE = 'frame_queue'

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Redis connection retry configuration
REDIS_RETRY_DELAY = 5  # seconds

class RedisConnectionManager:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = None

    def connect(self):
        while True:
            try:
                self.client = redis.Redis(host=self.host, port=self.port)
                self.client.ping()  # Test the connection
                logging.info("Successfully connected to Redis")
                return
            except redis.ConnectionError as e:
                logging.warning(f"Failed to connect to Redis: {str(e)}. Retrying in {REDIS_RETRY_DELAY} seconds...")
                time.sleep(REDIS_RETRY_DELAY)

    def get_client(self):
        if self.client is None:
            self.connect()
        return self.client

    def push_to_queue(self, queue, data):
        while True:
            try:
                self.get_client().rpush(queue, data)
                return
            except redis.ConnectionError as e:
                logging.warning(f"Redis connection lost: {str(e)}. Attempting to reconnect...")
                self.connect()  # Try to reconnect

# Initialize Redis connection manager
redis_manager = RedisConnectionManager(REDIS_HOST, REDIS_PORT)

def grab_frame(camera_url):
    """Attempt to grab a single frame from the given camera URL."""
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return frame

def grab_and_queue_frame(camera_id, camera_index):
    """Grab a single frame from the given camera URL and push it to the Redis queue."""
    camera_url = f"{BASE_URL}{camera_id}?enableSrtp"
    
    for attempt in range(MAX_RETRIES):
        frame = grab_frame(camera_url)
        if frame is not None:
            break
        logging.warning(f"Attempt {attempt + 1} failed for camera {camera_index}. Retrying...")
        time.sleep(RETRY_DELAY)
    
    if frame is None:
        logging.error(f"Failed to grab frame from camera {camera_index} after {MAX_RETRIES} attempts")
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
    redis_manager.push_to_queue(REDIS_QUEUE, str(frame_data))
    logging.info(f"Frame from camera {camera_index} pushed to Redis queue")

def main():
    # Ensure initial Redis connection
    redis_manager.connect()

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