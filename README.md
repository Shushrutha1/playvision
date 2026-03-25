![PlayVision](4.png)
#  PlayVision Analytics

**PlayVision Analytics** is an AI-powered smart player performance analyzer that processes match footage and generates performance insights using computer vision and machine learning.

The system detects players from uploaded match videos and analyzes their **movement, velocity, agility, and gameplay patterns**, helping coaches and analysts understand match dynamics.

---

##  Features

*  Upload match footage directly in the web interface
*  AI-powered player detection using YOLOv8
*  Velocity tracking of player movements
*  Agility and transition analysis
*  Automated AI match insights and performance reports
*  Premium dark UI dashboard built with Streamlit

---

##  Demo Workflow

1️⃣ Upload a match video
2️⃣ AI detects players frame-by-frame
3️⃣ Movement distance and speed are calculated
4️⃣ AI generates match analytics and commentary

---

##  Technology Stack

* **Python**
* **Streamlit**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **NumPy**
* **Computer Vision**
* **Machine Learning**

---

## 📂 Project Structure

```
playvision/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── runtime.txt         # Runtime configuration
├── yolov8n.pt          # Pre-trained YOLO model
├── 1.png               # App icon
├── 4.png               # Logo
```

---

##  Installation

Clone the repository:

```
git clone https://github.com/Shushrutha1/playvision.git
```

Navigate into the project folder:

```
cd playvision
```

Install required dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```
streamlit run app.py
```

The application will open automatically in your browser.

---

##  AI Performance Metrics

The system generates analytics including:

* **Runner Velocity**
* **Chaser Transition Speed**
* **Player Movement Distance**
* **AI Match Commentary**
* **Performance Accuracy Metrics**

---

##  Future Improvements

* Real-time player tracking dashboard
* Multi-player comparison analytics
* Advanced movement pattern detection
* Cloud deployment for live match analysis
* Mobile app integration for coaches

---


⭐ If you like this project, consider giving it a **star**!







