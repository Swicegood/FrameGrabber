import redis
import json
from datetime import datetime, timedelta

r = redis.Redis(host='192.168.0.71', port=6379, db=0)

def parse_frame_data(data):
    try:
        frame_data = eval(data.decode('utf-8'))
        return frame_data.get('timestamp')
    except:
        return None

def find_frames_in_time_range(start_time, end_time):
    results = []
    for i in range(r.llen('frame_queue')):
        item = r.lindex('frame_queue', i)
        timestamp = parse_frame_data(item)
        if timestamp:
            frame_time = datetime.fromisoformat(timestamp)
            if start_time <= frame_time <= end_time:
                results.append((i, timestamp))
    return results

# Example usage: Find frames from the last hour
now = datetime.now()
one_hour_ago = now - timedelta(hours=100)
frames = find_frames_in_time_range(one_hour_ago, now)

for index, timestamp in frames:
    print(f"Frame at index {index} with timestamp {timestamp}")