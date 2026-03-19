import cv2
import mediapipe as mp # type: ignore
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    movement = []
    prev_center = None
    active_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            active_frames += 1

            landmarks = result.pose_landmarks.landmark

            # Hip center for movement tracking
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            center = np.array([
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2
            ])

            if prev_center is not None:
                dist = np.linalg.norm(center - prev_center)
                movement.append(dist)

            prev_center = center

    cap.release()

    return movement, active_frames
