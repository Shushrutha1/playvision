import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import analyze_video

def calculate_kho_kho_performance(video_path):
    movement, active_frames = analyze_video(video_path)

    if len(movement) == 0:
        print("No player detected.")
        return

    speed = np.mean(movement) * 1000
    agility = np.std(movement) * 1000
    endurance = active_frames
    reaction = np.max(movement) * 1000

    performance_score = (
        speed * 0.3 +
        agility * 0.3 +
        endurance * 0.2 +
        reaction * 0.2
    )

    data = {
        "Speed": speed,
        "Agility": agility,
        "Endurance": endurance,
        "Reaction": reaction,
        "Performance Score": performance_score
    }

    df = pd.DataFrame([data])
    print("\n🏆 Kho-Kho Player Performance\n")
    print(df)

    df.plot(kind="bar", figsize=(8,4))
    plt.title("Kho-Kho Player Performance Metrics")
    plt.xticks(rotation=0)
    plt.show()

if __name__ == "__main__":
    calculate_kho_kho_performance("input_video.mp4")

