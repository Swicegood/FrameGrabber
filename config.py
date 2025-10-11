import os

BASE_URL = "rtsps://192.168.0.1:7441/"
AXIS_URL = "rtsp://jaga:ahare7462s@192.168.0.90/onvif-media/media.amp?profile=profile_1_h264&sessiontimeout=60&streamtype=unicast&fps=15&audio=1"

CAMERA_IDS = [
    "rHWz9GRDFxrOZF7b", "5SJZivf8PPsLWw2n", "g8rHNVCflWO1ptKN",
    "iY9STaEt7K9vS8yJ", "IOKAu7MMacLh79zn", "sHlS7ewuGDEd2ef4",
    "AXIS_ID", "prXH5H6e9GxOij1Z"
]

camera_names = {
    "rHWz9GRDFxrOZF7b": "Down_Pujari", "5SJZivf8PPsLWw2n": "Hall", "g8rHNVCflWO1ptKN": "Kitchen",
    "iY9STaEt7K9vS8yJ": "Prabhupada", "IOKAu7MMacLh79zn": "Temple", "sHlS7ewuGDEd2ef4": "Up_Pujari",
    "AXIS_ID": "Axis", "prXH5H6e9GxOij1Z": "Front_Driveway"
}

TOTAL_CAMERAS = len(CAMERA_IDS)
TARGET_FPS = float(os.getenv('FPS', "1/60").split('/')[0]) / float(os.getenv('FPS', "1/60").split('/')[1])
MAX_WORKERS = 20  # Adjusted from 44 for 8 cameras (was ~2.4 workers per camera)

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