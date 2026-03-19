import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# 1. Load Model
model = YOLO('yolov8n.pt')

# 2. Setup Video
video_path = r'C:\Users\pc\Desktop\SPAK\khokho_match.mp4'
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('khokho_role_analysis.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

def get_team_role(img_patch):
    """
    Analyzes the color of the player's jersey to determine team.
    You can adjust the color ranges based on your specific match video.
    """
    hsv = cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)
    
    # Example: Define range for Team A (e.g., Blue jerseys)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    if np.sum(mask) > 500: # If enough blue pixels are found
        return "CHASER", (0, 0, 255) # Red Box
    else:
        return "RUNNER", (0, 255, 0) # Green Box

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Higher confidence (0.5) ensures ONLY clear players are detected
    results = model.track(frame, persist=True, conf=0.5, classes=[0], tracker="botsort.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            # Crop the player from the frame to check jersey color
            x1, y1, x2, y2 = box
            player_crop = frame[y1:y2, x1:x2]
            
            if player_crop.size > 0:
                role, color = get_team_role(player_crop)

                # Draw Custom Bounding Box based on Role
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Fixed ID and Role Label
                label = f"ID:{track_id} | {role}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)
    cv2.imshow("Kho-Kho Role Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()