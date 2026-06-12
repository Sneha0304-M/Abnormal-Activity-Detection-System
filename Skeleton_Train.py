"""
Skeleton_Train.py  —  Standalone Pose Training Collector
---------------------------------------------------------
Loads existing CSV on startup (persistent).
Press S to save current pose with a label.
Press Q to quit.
Data is always appended — never overwritten.
"""

from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd
import os

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
cap        = cv2.VideoCapture(0)

PATH_TO_SAVE = "activity.csv"

LANDMARK_NAMES = [
    'nose','left_eye_inner','left_eye','left_eye_outer',
    'right_eye_inner','right_eye','right_eye_outer',
    'left_ear','right_ear','mouth_left','mouth_right',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_pinky_1','right_pinky_1',
    'left_index_1','right_index_1','left_thumb_2','right_thumb_2',
    'left_hip','right_hip','left_knee','right_knee',
    'left_ankle','right_ankle','left_heel','right_heel',
    'left_foot_index','right_foot_index',
]

col_names = []
for name in LANDMARK_NAMES:
    col_names.extend([name+'_X', name+'_Y', name+'_Z', name+'_V'])

# ── Load existing CSV (persistent) ──
full_lm_list = []
target_list  = []

if os.path.exists(PATH_TO_SAVE):
    df_ex = pd.read_csv(PATH_TO_SAVE)
    for _, row in df_ex.iterrows():
        full_lm_list.append(row.iloc[:132].values.tolist())
        target_list.append(row['Pose_Class'])
    print(f"[CSV] Loaded {len(full_lm_list)} existing samples.")
else:
    print("[CSV] No existing CSV found — starting fresh.")


@app.route('/')
def index():
    return render_template('index.html')


def _gen():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        RGB    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(RGB)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Samples: {len(full_lm_list)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/train', methods=['POST'])
def train():
    try:
        data       = request.get_json()
        class_name = data['class_name'].strip()

        _, frame = cap.read()
        RGB      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result   = pose.process(RGB)

        if not result.pose_landmarks:
            return jsonify({"message": "No landmarks detected!"}), 400

        lm_list = list(result.pose_landmarks.landmark)

        cx = (lm_list[LANDMARK_NAMES.index('right_hip')].x +
              lm_list[LANDMARK_NAMES.index('left_hip')].x) * 0.5
        cy = (lm_list[LANDMARK_NAMES.index('right_hip')].y +
              lm_list[LANDMARK_NAMES.index('left_hip')].y) * 0.5
        sx = (lm_list[LANDMARK_NAMES.index('right_shoulder')].x +
              lm_list[LANDMARK_NAMES.index('left_shoulder')].x) * 0.5
        sy = (lm_list[LANDMARK_NAMES.index('right_shoulder')].y +
              lm_list[LANDMARK_NAMES.index('left_shoulder')].y) * 0.5

        max_d = max(
            math.sqrt((lm.x - cx)**2 + (lm.y - cy)**2)
            for lm in lm_list
        )
        torso = math.sqrt((sx - cx)**2 + (sy - cy)**2)
        max_d = max(torso * 2.5, max_d)

        pre_lm = np.array([
            [(lm.x - cx)/max_d, (lm.y - cy)/max_d,
             lm.z/max_d, lm.visibility]
            for lm in lm_list
        ]).flatten().tolist()

        full_lm_list.append(pre_lm)
        target_list.append(class_name)

        # Append to existing CSV — re-write from the in-memory list
        # (which was loaded from CSV on startup, so nothing is lost)
        df = pd.DataFrame(full_lm_list, columns=col_names)
        df['Pose_Class'] = target_list
        df.to_csv(PATH_TO_SAVE, encoding='utf-8', index=False)

        return jsonify({
            "message": f"'{class_name}' saved! Total: {len(full_lm_list)}"
        })

    except Exception as exc:
        return jsonify({"message": f"Error: {exc}"}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5001)
