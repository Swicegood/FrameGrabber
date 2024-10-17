import time
import logging
from concurrent.futures import ThreadPoolExecutor
from config import CAMERA_IDS, camera_names, MAX_WORKERS, TARGET_FPS
from redis_manager import clear_redis_data
from camera_operations import grab_and_store_frame
from image_processing import update_composite_images

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    clear_redis_data(CAMERA_IDS, camera_names)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        last_composite_update = time.time()
        while True:
            start_time = time.time()
            
            futures = [executor.submit(grab_and_store_frame, camera_id, i+1, camera_names) 
                       for i, camera_id in enumerate(CAMERA_IDS)]
            
            for future in futures:
                future.result()
            
            if time.time() - last_composite_update >= 60:
                update_composite_images(CAMERA_IDS, camera_names)
                last_composite_update = time.time()
            
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1 / TARGET_FPS) - elapsed_time) if TARGET_FPS else 0
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()