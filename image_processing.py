import cv2
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from config import HOURLY_FRAMES_KEY, COMPOSITE_IMAGE_KEY

def generate_composite_image(camera_id, camera_names, redis_client):
    hourly_key = HOURLY_FRAMES_KEY.format(camera_names[camera_id])
    frames = redis_client.lrange(hourly_key, 0, -1)
    
    if not frames:
        return None
    
    base_frame = None
    change_accumulator = None
    prev_frame = None
    included_frames = []
    
    for frame_data in frames:
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if base_frame is None:
            base_frame = img.copy().astype(float)
            change_accumulator = np.zeros(img.shape[:2], dtype=np.float32)
            prev_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            included_frames.append(img)
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(prev_frame, gray)
        
        if ssim_value < 0.95:
            cv2.accumulateWeighted(img, base_frame, 0.1)
            frame_diff = cv2.absdiff(gray, prev_frame)
            thresh = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            change_accumulator += thresh.astype(np.float32) / 255.0
            included_frames.append(img)
        
        prev_frame = gray
    
    base_frame = cv2.normalize(base_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    max_change = np.max(change_accumulator)
    if max_change > 0:
        change_accumulator /= max_change
    
    color_overlay = np.zeros_like(base_frame)
    color_overlay[..., 2] = (change_accumulator * 255).astype(np.uint8)
    
    color_overlay = cv2.GaussianBlur(color_overlay, (5, 5), 0)
    
    result = cv2.addWeighted(base_frame, 0.7, color_overlay, 0.3, 0)
    
    legend_height = 30
    legend = np.zeros((legend_height, result.shape[1], 3), dtype=np.uint8)
    for i in range(result.shape[1]):
        legend[:, i] = [0, 0, int(255 * i / result.shape[1])]
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(result, timestamp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(legend, 'Low Activity', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(legend, 'High Activity', (result.shape[1] - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    result = np.vstack((result, legend))

    cv2.putText(result, f'Frames: {len(included_frames)}/{len(frames)}', (10, result.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    _, buffer = cv2.imencode('.png', result)
    return buffer.tobytes()

def update_composite_images(camera_ids, camera_names, redis_client):
    for camera_id in camera_ids:
        composite = generate_composite_image(camera_id, camera_names, redis_client)
        if composite:
            composite_key = COMPOSITE_IMAGE_KEY.format(camera_names[camera_id])
            redis_client.set(composite_key, composite)
            logging.info(f"Updated composite image for camera {camera_names[camera_id]}")