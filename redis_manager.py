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

    def execute_with_retry(self, method, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return method(*args, **kwargs)
            except (redis.ConnectionError, ConnectionRefusedError) as e:
                logging.warning(f"Redis operation failed: {str(e)}. Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    self.connect()  # Try to reconnect
                else:
                    logging.error("Max retries reached. Redis operation failed.")
                    raise

    def lpush(self, *args, **kwargs):
        return self.execute_with_retry(self.get_client().lpush, *args, **kwargs)

    def set(self, *args, **kwargs):
        return self.execute_with_retry(self.get_client().set, *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.execute_with_retry(self.get_client().delete, *args, **kwargs)

    def ltrim(self, *args, **kwargs):
        return self.execute_with_retry(self.get_client().ltrim, *args, **kwargs)
    
    def lrange(self, *args, **kwargs):
        return self.execute_with_retry(self.get_client().lrange, *args, **kwargs)

redis_manager = RedisConnectionManager(REDIS_HOST, REDIS_PORT)

def clear_redis_data(camera_ids, camera_names):
    from config import REDIS_QUEUE, HOURLY_FRAMES_KEY, COMPOSITE_IMAGE_KEY
    
    redis_manager.delete(REDIS_QUEUE)
    
    for camera_id in camera_ids:
        camera_name = camera_names[camera_id]
        hourly_key = HOURLY_FRAMES_KEY.format(camera_name)
        composite_key = COMPOSITE_IMAGE_KEY.format(camera_name)
        redis_manager.delete(hourly_key)
        redis_manager.delete(composite_key)
    
    logging.info("Cleared all relevant Redis keys")