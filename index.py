import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load Model
model = YOLO('yolov8n.pt')

# Load Video
video_path = r'C:\Users\pc\Desktop\SPAK\khokho_match.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Metrics Storage
stats = defaultdict(lambda: {'positions': [], 'total_dist': 0})
team_metrics = {'Chasers': [], 'Runners': []}

print("🚀 Analyzing match performance... please wait.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True, conf=0.4, classes=[0], verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            
            # Record movement
            if len(stats[track_id]['positions']) > 0:
                prev_x, prev_y = stats[track_id]['positions'][-1]
                dist = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                stats[track_id]['total_dist'] += dist
            
            stats[track_id]['positions'].append((cx, cy))

cap.release()

# --- PERFORMANCE CALCULATION ---
print("\n" + "="*30)
print("   KHO-KHO PERFORMANCE REPORT")
print("="*30)

total_speeds = []
all_accuracies = []

for p_id, data in stats.items():
    if len(data['positions']) < 10: continue # Skip spectators/glitches
    
    # 1. Calculate Speed (Pixels per frame converted to a relative scale)
    avg_speed = (data['total_dist'] / len(data['positions'])) * (fps / 10) 
    total_speeds.append(avg_speed)

    # 2. Calculate Accuracy (Path Efficiency)
    # Efficiency = Distance between Start/End / Total path distance
    start = np.array(data['positions'][0])
    end = np.array(data['positions'][-1])
    displacement = np.linalg.norm(end - start)
    path_accuracy = (displacement / data['total_dist'] * 100) if data['total_dist'] > 0 else 0
    all_accuracies.append(path_accuracy)

    print(f"PLAYER ID: {p_id}")
    print(f" - Speed: {avg_speed:.2f} units/s")
    print(f" - Accuracy: {path_accuracy:.1f}%")
    print("-" * 20)

# --- CONCLUSION LOGIC ---
avg_team_speed = np.mean(total_speeds) if total_speeds else 0
avg_team_acc = np.mean(all_accuracies) if all_accuracies else 0

print("\nFINAL PERFORMANCE SUMMARY:")
print(f"Average Team Speed    : {avg_team_speed:.2f}")
print(f"Average Team Accuracy : {avg_team_acc:.1f}%")

# Conclusion Line Generation
if avg_team_speed > 5 and avg_team_acc > 70:
    conclusion = "High athletic performance: The team showed explosive speed with highly disciplined directional movement."
elif avg_team_speed > 5:
    conclusion = "Good speed but low coordination: Players are fast but movement patterns are erratic (high stamina, low tactics)."
else:
    conclusion = "Moderate performance: Consistent pacing observed, but requires improvement in rapid acceleration and path optimization."

print(f"\nCONCLUSION: {conclusion}")
print("="*30)



