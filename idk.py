import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os

# 1. Force CPU and Nano Model (Fastest)
model = YOLO('yolov8n.pt') 

# Use a relative path if the file is in the same folder, or double-check this path
video_path = r'C:\Users\pc\Desktop\SPAK\khokho_match.mp4'

if not os.path.exists(video_path):
    print(f"❌ Error: Video file not found at {video_path}")
    exit()

stats = defaultdict(lambda: {'pos': [], 'dist': 0})
start_time = time.time()

print("🚀 Starting Ultra-Fast Analysis...")

# 2. Optimized Inference Settings
# vid_stride=30: Processes only 1 frame per second (Massive speedup)
# imgsz=160: Tiny image size for the AI to process instantly
# stream=True: Processes one frame at a time to prevent memory freeze
results = model.track(
    source=video_path, 
    stream=True, 
    persist=True, 
    conf=0.5, 
    imgsz=160, 
    classes=[0], 
    tracker="bytetrack.yaml", 
    verbose=False,
    vid_stride=30,  
    device='cpu' 
)

# 3. Fast Data Collection Loop
for r in results:
    if r.boxes.id is not None:
        boxes = r.boxes.xywh.cpu().numpy() 
        ids = r.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            cx, cy = box[0], box[1]
            p_data = stats[track_id]
            
            if p_data['pos']:
                prev_x, prev_y = p_data['pos'][-1]
                # Manhattan distance is slightly faster than Euclidean for quick checks
                dist = abs(cx - prev_x) + abs(cy - prev_y)
                p_data['dist'] += dist
            
            p_data['pos'].append((cx, cy))

# --- 4. TEAM PERFORMANCE AGGREGATION (CHASERS VS RUNNERS) ---
end_time = time.time()
print(f"\n✅ Analysis Complete! Total Time: {end_time - start_time:.2f}s")
print("="*45)
print(" 🛡️  SPAK: KHO-KHO MATCH ANALYSIS")
print("="*45)

chasers_speed, chasers_acc = [], []
runners_speed, runners_acc = [], []

for p_id, data in stats.items():
    if len(data['pos']) > 3:
        avg_speed = data['dist'] / len(data['pos'])
        
        # Calculate Accuracy
        start_pt = np.array(data['pos'][0])
        end_pt = np.array(data['pos'][-1])
        displacement = np.linalg.norm(end_pt - start_pt)
        acc = (displacement / data['dist'] * 100) if data['dist'] > 0 else 0
        
        # LOGIC: Runners usually move more and have higher average speeds 
        # than crouching Chasers. (Adjust the threshold '30' based on your results)
        if avg_speed > 30: 
            runners_speed.append(avg_speed)
            runners_acc.append(acc)
        else:
            chasers_speed.append(avg_speed)
            chasers_acc.append(acc)

# --- CONCLUSION LOGIC ---
def get_team_report(speeds, accs, team_type):
    if not speeds: return f"No {team_type} activity detected."
    
    avg_s = np.mean(speeds)
    avg_a = np.mean(accs)
    
    if team_type == "Runner":
        if avg_s > 50:
            return "✅ The Runner team is good... but they need to improve their path accuracy to dodge better."
        else:
            return "⚠️ The Runner team needs to improve their speed; they are currently too easy to catch."
            
    if team_type == "Chaser":
        if avg_s > 20 and avg_a > 70:
            return "🔥 The Chaser team is very good! Their quick reactions will be useful for the last moment."
        else:
            return "📉 The Chaser team is slow to react; they need faster 'Kho' transitions."

# Print Final Team Insights
print(f"🏃 RUNNER STATS: Avg Speed {np.mean(runners_speed) if runners_speed else 0:.1f}")
print(f"🧍 CHASER STATS: Avg Speed {np.mean(chasers_speed) if chasers_speed else 0:.1f}")
print("-" * 45)
print(f"RUNNER REPORT: {get_team_report(runners_speed, runners_acc, 'Runner')}")
print(f"CHASER REPORT: {get_team_report(chasers_speed, chasers_acc, 'Chaser')}")
print("="*45)