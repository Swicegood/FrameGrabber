import cv2
import time
import logging
from datetime import datetime
from config import BASE_URL, AXIS_URL, MAX_RETRIES, RETRY_DELAY, REDIS_FRAME_KEY, HOURLY_FRAMES_KEY

def get_camera_url(camera_id):
    if camera_id == "AXIS_ID":
        return AXIS_URL
    else:
        return f"{BASE_URL}{camera_id}?enableSrtp"

def grab_frame(camera_url):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return frame

def grab_and_store_frame(camera_id, camera_index, camera_names, redis_client):
    camera_url = get_camera_url(camera_id)
    
    for attempt in range(MAX_RETRIES):
        frame = grab_frame(camera_url)
        if frame is not None:
            break
        logging.warning(f"Attempt {attempt + 1} failed for camera {camera_index}. Retrying...")
        time.sleep(RETRY_DELAY)
        
    if frame is None:
        logging.error(f"Failed to grab frame from camera {camera_index} after {MAX_RETRIES} attempts")
        return
    
    _, buffer = cv2.imencode('.png', frame)
    png_as_text = buffer.tobytes()
    
    frame_data = {
        'camera_name': camera_names[camera_id],
        'camera_id': camera_id,
        'camera_index': camera_index,
        'timestamp': datetime.now().isoformat(),
        'frame': png_as_text  
    }
    
    hourly_key = HOURLY_FRAMES_KEY.format(camera_names[camera_id])
    redis_client.lpush(hourly_key, png_as_text)
    redis_client.ltrim(hourly_key, 0, 59)
    redis_client.set(REDIS_FRAME_KEY.format(camera_id), str(frame_data))
    
    logging.info(f"Stored new frame for camera {camera_index}")