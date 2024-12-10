import os

BASE_URL = "rtsps://192.168.0.1:7441/"
AXIS_URL = "rtsp://jaga:ahare7462s@192.168.0.90/onvif-media/media.amp?profile=profile_1_h264&sessiontimeout=60&streamtype=unicast&fps=15&audio=1"

CAMERA_IDS = [
    "I6Dvhhu1azyV9rCu", "PxnDZaXu2awYbMmS", "oaQllpjP0sk94nCV",
    "mKlJgNx7tXwalch1", "rHWz9GRDFxrOZF7b", "LRqgKMMjjJbNEeyE",
    "94uZsJ2yIouIXp2x", "5SJZivf8PPsLWw2n", "g8rHNVCflWO1ptKN",
    "t3ZIWTl9jZU1JGEI", "iY9STaEt7K9vS8yJ", "jlNNdFFvhQ2o2kmn",
    "IOKAu7MMacLh79zn", "sHlS7ewuGDEd2ef4", "OSF13XTCKhpIkyXc",
    "jLUEC60zHGo7BXfj", "AXIS_ID", "prXH5H6e9GxOij1Z"
]

camera_names = {
    "I6Dvhhu1azyV9rCu": "Audio_Visual", "oaQllpjP0sk94nCV": "Bhoga_Shed", "PxnDZaXu2awYbMmS": "Back_Driveway",
    "mKlJgNx7tXwalch1": "Deck_Stairs", "rHWz9GRDFxrOZF7b": "Down_Pujari", "LRqgKMMjjJbNEeyE": "Field",
    "94uZsJ2yIouIXp2x": "Greenhouse", "5SJZivf8PPsLWw2n": "Hall", "g8rHNVCflWO1ptKN": "Kitchen",
    "t3ZIWTl9jZU1JGEI": "Pavillion", "iY9STaEt7K9vS8yJ": "Prabhupada", "jlNNdFFvhQ2o2kmn": "Stage",
    "IOKAu7MMacLh79zn": "Temple", "sHlS7ewuGDEd2ef4": "Up_Pujari", "OSF13XTCKhpIkyXc": "Walk-in",
    "jLUEC60zHGo7BXfj": "Walkway", "AXIS_ID": "Axis", "prXH5H6e9GxOij1Z": "Front Driveway"  # Add the new camera name here
}

TOTAL_CAMERAS = len(CAMERA_IDS)
TARGET_FPS = float(os.getenv('FPS', "1/60").split('/')[0]) / float(os.getenv('FPS', "1/60").split('/')[1])
MAX_WORKERS = 44

REDIS_HOST = '192.168.0.71'
REDIS_PORT = 6379
REDIS_QUEUE = 'frame_queue'
REDIS_FRAME_KEY = "camera_frames:{}"
HOURLY_FRAMES_KEY = 'hourly_frames_{}'
COMPOSITE_IMAGE_KEY = 'composite_{}'

MAX_RETRIES = 3
RETRY_DELAY = 1
REDIS_RETRY_DELAY = 5
EXPIRATION_TIME = 300