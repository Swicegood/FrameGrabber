import time
import cv2
import redis
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import os
import io

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

camera_names = {
    "I6Dvhhu1azyV9rCu": "Audio_Visual", "oaQllpjP0sk94nCV": "Bhoga_Shed", "PxnDZaXu2awYbMmS": "Back_Driveway",
    "mKlJgNx7tXwalch1": "Deck_Stairs", "rHWz9GRDFxrOZF7b": "Down_Pujari", "LRqgKMMjjJbNEeyE": "Field",
    "94uZsJ2yIouIXp2x": "Greenhouse", "5SJZivf8PPsLWw2n": "Hall", "g8rHNVCflWO1ptKN": "Kitchen",
    "t3ZIWTl9jZU1JGEI": "Pavillion", "iY9STaEt7K9vS8yJ": "Prabhupada", "jlNNdFFvhQ2o2kmn": "Stage",
    "IOKAu7MMacLh79zn": "Temple", "sHlS7ewuGDEd2ef4": "Up_Pujari", "OSF13XTCKhpIkyXc": "Walk-in",
    "jLUEC60zHGo7BXfj": "Walkway"
}

TOTAL_CAMERAS = len(CAMERA_IDS)
TARGET_FPS = os.getenv('FPS', "1/60")  # frame rate
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


# Add new Redis keys for storing hourly frames and composite images
HOURLY_FRAMES_KEY = 'hourly_frames_{}'  # Will be formatted with camera_id
COMPOSITE_IMAGE_KEY = 'composite_{}'  # Will be formatted with camera_id

# Assuming TARGET_FPS is a string like '1/180'
# This function safely evaluates simple mathematical expressions for division
def safe_eval_fraction(expr):
    try:
        numerator, denominator = expr.split('/')
        return float(numerator) / float(denominator)
    except ValueError:
        # Return None or raise an error if the expression is not a simple fraction
        return None

# Use the safe_eval_fraction function to convert TARGET_FPS to a float
TARGET_FPS = safe_eval_fraction(TARGET_FPS)


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
    """Grab a single frame from the given camera URL, push it to the Redis queue, and store for composite."""
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
        'camera_name': camera_names[camera_id],
        'camera_id': camera_id,
        'camera_index': camera_index,
        'timestamp': datetime.now().isoformat(),
        'frame': png_as_text
    }
    
    # Push to Redis queue for LLM processing
    redis_manager.push_to_queue(REDIS_QUEUE, str(frame_data))
    
    # Store frame for hourly composite
    hourly_key = HOURLY_FRAMES_KEY.format(camera_names[camera_id])
    redis_manager.get_client().lpush(hourly_key, png_as_text)
    redis_manager.get_client().ltrim(hourly_key, 0, 59)  # Keep only the last 60 frames (1 hour at 1 frame per minute)
    
    logging.info(f"Frame from camera {camera_index} pushed to Redis queue and stored for hourly composite")

def generate_composite_image(camera_name):
    """Generate a composite image from the last hour of frames."""
    hourly_key = HOURLY_FRAMES_KEY.format(camera_name)
    frames = redis_manager.get_client().lrange(hourly_key, 0, -1)
    
    if not frames:
        return None
    
    composite = None
    for frame_data in frames:
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if composite is None:
            composite = np.zeros_like(img, dtype=np.float32)
        
        composite += img.astype(np.float32) / len(frames)
    
    composite = composite.astype(np.uint8)
    _, buffer = cv2.imencode('.png', composite)
    return buffer.tobytes()

def update_composite_images():
    """Update composite images for all cameras."""
    for camera_id in CAMERA_IDS:
        composite = generate_composite_image(camera_names[camera_id])
        if composite:
            composite_key = COMPOSITE_IMAGE_KEY.format(camera_names[camera_id])
            redis_manager.get_client().set(composite_key, composite)
            logging.info(f"Updated composite image for camera {camera_names[camera_id]}")

def main():
    # Ensure initial Redis connection
    redis_manager.connect()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        last_composite_update = time.time()
        while True:
            start_time = time.time()
            
            # Submit tasks for all cameras
            futures = [executor.submit(grab_and_queue_frame, camera_id, i+1) 
                       for i, camera_id in enumerate(CAMERA_IDS)]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
            
            # Update composite images every minute
            if time.time() - last_composite_update >= 60:
                update_composite_images()
                last_composite_update = time.time()
            
            # Calculate sleep time to maintain target FPS
            elapsed_time = time.time() - start_time
            if TARGET_FPS:  # Ensure TARGET_FPS is not None or 0 to avoid division by zero
                sleep_time = max(0, (1 / TARGET_FPS) - elapsed_time)
            else:
                sleep_time = 0
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()