# =============================================================================
#  app.py  —  Flask Application
# =============================================================================
#  FIXES vs previous version:
#  1. Training video feed (_gen_simple) now properly shows multiple coloured
#     skeletons — uses the same bg_sub + ROI pipeline as detection, with a
#     full-frame fallback until bg_sub warms up (first 30 frames).
#  2. /alert_action no longer triggers email (email is now sent automatically
#     on detection in Videocam.py).  Endpoint only updates UI status.
#  3. /train — persistent CSV append unchanged.
#  4. All other routes unchanged.
# =============================================================================

from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd
import os

from Videocam import (
    VideoCamera,
    pending_alerts, pending_alerts_lock,
    LANDMARK_NAMES, CSV_PATH,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
#  Streams
# ---------------------------------------------------------------------------
video_stream = VideoCamera()

mp_drawing   = mp.solutions.drawing_utils
mp_pose      = mp.solutions.pose

# Separate camera instance for training page
# (detection stream and training stream share the same physical camera
#  via VideoCapture — only ONE can hold the device at a time.
#  We reuse the VideoCamera's cap for training snapshots.)
_pose_train  = mp_pose.Pose(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5,
                              model_complexity=1)

# Training feed uses its own cv2.VideoCapture so it doesn't conflict
_train_cap   = cv2.VideoCapture(0)

PERSON_COLORS = [
    (255, 100,  50),
    ( 50, 220,  50),
    ( 50,  80, 255),
    (220, 200,  30),
]

col_names = []
for name in LANDMARK_NAMES:
    col_names.extend([name+'_X', name+'_Y', name+'_Z', name+'_V'])


# ===========================================================================
#  PAGE ROUTES
# ===========================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect')
def detect():
    return render_template('Abnormal_Activity.html')


# ===========================================================================
#  TRAINING VIDEO FEED  — shows multiple coloured skeletons
#  Uses bg subtraction to isolate each person, with full-frame fallback.
# ===========================================================================
def _gen_simple():
    MAX       = 4
    pose_pool = [
        mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1          # same quality as detection
        )
        for _ in range(MAX)
    ]
    bg         = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )
    font       = cv2.FONT_HERSHEY_SIMPLEX
    frame_cnt  = 0

    def _get_boxes(frame):
        h, w    = frame.shape[:2]
        mask    = bg.apply(frame)
        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
        mask    = cv2.dilate(mask, k, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if (h*w*0.004) < area < (h*w*0.95):
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bh > bw * 0.35:
                    px = int(bw*0.20); py = int(bh*0.10)
                    boxes.append((
                        max(0, x-px), max(0, y-py),
                        min(w, x+bw+px)-max(0, x-px),
                        min(h, y+bh+py)-max(0, y-py)
                    ))
        return boxes[:MAX]

    def _draw_skel(frame, landmarks, color, h, w):
        for conn in mp_pose.POSE_CONNECTIONS:
            s = landmarks.landmark[conn[0]]
            e = landmarks.landmark[conn[1]]
            if s.visibility > 0.4 and e.visibility > 0.4:
                cv2.line(frame,
                         (int(s.x*w), int(s.y*h)),
                         (int(e.x*w), int(e.y*h)),
                         (30,30,30), 4)
                cv2.line(frame,
                         (int(s.x*w), int(s.y*h)),
                         (int(e.x*w), int(e.y*h)),
                         color, 2)
        for lm in landmarks.landmark:
            if lm.visibility > 0.4:
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (cx,cy), 6, (255,255,255), -1)
                cv2.circle(frame, (cx,cy), 4, color, -1)

    while _train_cap.isOpened():
        ret, frame = _train_cap.read()
        if not ret:
            break
        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        RGB       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cnt += 1

        # Before bg_sub warms up, use full-frame single pose
        boxes    = _get_boxes(frame) if frame_cnt > 30 else []
        detected = 0

        if not boxes:
            # Full-frame fallback
            res = pose_pool[0].process(RGB)
            if res and res.pose_landmarks:
                color = PERSON_COLORS[0]
                _draw_skel(frame, res.pose_landmarks, color, h, w)
                cv2.putText(frame, "P1", (10, 30), font, 0.7, color, 2, cv2.LINE_AA)
                detected = 1
        else:
            for idx, (bx, by, bw2, bh2) in enumerate(boxes):
                color = PERSON_COLORS[idx % len(PERSON_COLORS)]
                roi   = RGB[by:by+bh2, bx:bx+bw2]
                if roi.size == 0:
                    continue
                res = pose_pool[idx].process(roi)
                if res and res.pose_landmarks:
                    # Remap to full-frame coords
                    for lm in res.pose_landmarks.landmark:
                        lm.x = (lm.x * bw2 + bx) / w
                        lm.y = (lm.y * bh2 + by) / h
                    _draw_skel(frame, res.pose_landmarks, color, h, w)
                    cv2.putText(frame, f"P{idx+1}",
                                (bx+5, max(by-8, 18)), font, 0.65,
                                color, 2, cv2.LINE_AA)
                    detected += 1

        # HUD
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (220, 32), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Persons: {detected}",
                    (8, 22), font, 0.65, (0,255,100), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def _gen_detect(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(_gen_simple(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_act')
def detect_act():
    return Response(_gen_detect(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ===========================================================================
#  TRAIN  —  PERSISTENT CSV (append-only)
# ===========================================================================
@app.route('/train', methods=['POST'])
def train():
    try:
        data       = request.get_json()
        class_name = data['class_name'].strip()
        if not class_name:
            return jsonify({"message": "Class name cannot be empty"}), 400

        # Capture from the training feed camera
        _, frame = _train_cap.read()
        if frame is None:
            return jsonify({"message": "Camera not available"}), 500

        frame  = cv2.flip(frame, 1)
        RGB    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _pose_train.process(RGB)

        if not result.pose_landmarks:
            return jsonify({"message": "No landmarks detected — stand fully in frame"}), 400

        lm_list = list(result.pose_landmarks.landmark)

        # Normalise — same formula as classify_pose
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
            [(lm.x - cx) / max_d,
             (lm.y - cy) / max_d,
             lm.z        / max_d,
             lm.visibility]
            for lm in lm_list
        ]).flatten()

        # ── Persistent append ──────────────────────────────────────────
        if os.path.isfile(CSV_PATH):
            existing = pd.read_csv(CSV_PATH)
        else:
            existing = pd.DataFrame(columns=col_names + ['Pose_Class'])

        new_row               = pd.DataFrame([pre_lm], columns=col_names)
        new_row['Pose_Class'] = class_name
        updated               = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(CSV_PATH, encoding='utf-8', index=False)
        # ──────────────────────────────────────────────────────────────

        # Reload in VideoCamera so detection picks it up immediately
        video_stream.reload_csv()

        total = len(updated)
        return jsonify({
            "message": f"'{class_name}' saved! Total samples: {total}"
        })

    except Exception as exc:
        return jsonify({"message": f"Error: {exc}"}), 500


# ===========================================================================
#  RELOAD CSV
# ===========================================================================
@app.route('/reload_csv', methods=['POST'])
def reload_csv():
    try:
        video_stream.reload_csv()
        n = len(video_stream._df_labels)
        return jsonify({"message": f"Reloaded — {n} samples active."})
    except Exception as exc:
        return jsonify({"message": f"Error: {exc}"}), 500


# ===========================================================================
#  ALERT POLLING  (browser polls every 1.5 s)
# ===========================================================================
@app.route('/get_alerts')
def get_alerts():
    with pending_alerts_lock:
        result = [
            {
                "id":        a["id"],
                "person_id": a["person_id"],
                "label":     a["label"],
                "status":    a["status"],
            }
            for a in pending_alerts
            if a["status"] == "pending"
        ]
    return jsonify(result)


# ===========================================================================
#  ALERT ACTION  — only updates UI status (email already sent automatically)
# ===========================================================================
@app.route('/alert_action', methods=['POST'])
def alert_action():
    data     = request.get_json()
    alert_id = data.get('id')
    action   = data.get('action')   # "dismissed"

    with pending_alerts_lock:
        for a in pending_alerts:
            if a["id"] == alert_id and a["status"] == "pending":
                a["status"] = action
                return jsonify({
                    "message": f"Alert {action}. (Email was already sent automatically.)"
                })

    return jsonify({"message": "Alert not found or already actioned."}), 404


# ===========================================================================
#  ENTRY POINT
# ===========================================================================
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
