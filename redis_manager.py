import redis
import logging
import time
from config import REDIS_HOST, REDIS_PORT, REDIS_RETRY_DELAY

class RedisConnectionManager:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = None

    def connect(self):
        while True:
            try:
                self.client = redis.Redis(host=self.host, port=self.port)
                self.client.ping()
                logging.info("Successfully connected to Redis")
                return
            except redis.ConnectionError as e:
               logging.warning(f"Failed to connect to Redis: {str(e)}. Retrying in {REDIS_RETRY_DELAY} seconds...")
               time.sleep(REDIS_RETRY_DELAY)

    def get_client(self):
        if self.client is None:
            self.connect()
        return self.client

redis_manager = RedisConnectionManager(REDIS_HOST, REDIS_PORT)

def clear_redis_data(redis_client, camera_ids, camera_names):
    from config import REDIS_QUEUE, HOURLY_FRAMES_KEY, COMPOSITE_IMAGE_KEY
    
    redis_client.delete(REDIS_QUEUE)
    
    for camera_id in camera_ids:
        camera_name = camera_names[camera_id]
        hourly_key = HOURLY_FRAMES_KEY.format(camera_name)
        composite_key = COMPOSITE_IMAGE_KEY.format(camera_name)
        redis_client.delete(hourly_key)
        redis_client.delete(composite_key)
    
    logging.info("Cleared all relevant Redis keys")