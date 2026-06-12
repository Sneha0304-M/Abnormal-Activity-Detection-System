"""
Skeleton.py  —  Multi-Person Skeleton Tracker (Standalone)
Detects multiple persons per frame using background subtraction,
runs MediaPipe Pose on each person's ROI, and draws colored skeletons.
"""

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Colors for each detected person (BGR)
PERSON_COLORS = [
    (255, 0,   0),    # Blue
    (0,   255, 0),    # Green
    (0,   0,   255),  # Red
    (255, 255, 0),    # Cyan
    (255, 0,   255),  # Magenta
    (0,   255, 255),  # Yellow
]

MAX_PERSONS = 6

# Create a pool of pose estimators — one per person slot
pose_pool = [
    mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    for _ in range(MAX_PERSONS)
]

# Background subtractor for person detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Multi-Person Skeleton Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multi-Person Skeleton Tracker", 960, 540)

font = cv2.FONT_HERSHEY_SIMPLEX


def get_person_boxes(frame, max_persons):
    """Detect person bounding boxes via background subtraction."""
    h, w = frame.shape[:2]

    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (h * w * 0.005) < area < (h * w * 0.95):
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh > bw * 0.4:  # person-like aspect ratio
                pad_x = int(bw * 0.25)
                pad_y = int(bh * 0.15)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                boxes.append((x1, y1, x2 - x1, y2 - y1))

    # Sort by area (largest first) and limit
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_persons]


def draw_skeleton(frame, landmarks, color, h, w):
    """Draw colored skeleton on the frame."""
    for conn in mp_pose.POSE_CONNECTIONS:
        start = landmarks.landmark[conn[0]]
        end = landmarks.landmark[conn[1]]
        if start.visibility > 0.5 and end.visibility > 0.5:
            sx, sy = int(start.x * w), int(start.y * h)
            ex, ey = int(end.x * w), int(end.y * h)
            cv2.line(frame, (sx, sy), (ex, ey), color, 2)

    for lm in landmarks.landmark:
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = get_person_boxes(frame, MAX_PERSONS)
    detected = 0

    if not boxes:
        # Fallback: run pose on full frame
        result = pose_pool[0].process(RGB)
        if result.pose_landmarks:
            draw_skeleton(frame, result.pose_landmarks, PERSON_COLORS[0], h, w)
            cv2.putText(frame, "Person 1", (10, 35), font, 0.8, PERSON_COLORS[0], 2)
            detected = 1
    else:
        for idx, (bx, by, bw2, bh2) in enumerate(boxes):
            color = PERSON_COLORS[idx % len(PERSON_COLORS)]

            # Draw bounding box
            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), color, 2)

            # Run pose on ROI
            roi = RGB[by:by + bh2, bx:bx + bw2]
            if roi.size == 0:
                continue

            result = pose_pool[idx].process(roi)
            if result and result.pose_landmarks:
                # Re-map ROI coordinates to full frame
                for lm in result.pose_landmarks.landmark:
                    lm.x = (lm.x * bw2 + bx) / w
                    lm.y = (lm.y * bh2 + by) / h

                draw_skeleton(frame, result.pose_landmarks, color, h, w)
                label_y = max(by - 10, 20)
                cv2.putText(frame, f"Person {idx + 1}", (bx + 4, label_y),
                            font, 0.75, color, 2, cv2.LINE_AA)
                detected += 1

    # HUD
    cv2.putText(frame, f"Detected: {detected} person(s)", (10, h - 15),
                font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit", (w - 180, h - 15),
                font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Multi-Person Skeleton Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
for pose in pose_pool:
    pose.close()
cv2.destroyAllWindows()
