import streamlit as st
import cv2
import numpy as np
import math

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align:center; margin-bottom:1rem; }
    .gesture-display { font-size: 2rem; color: #ff4b4b; text-align:center; padding:10px;
                       border:2px solid #ff4b4b; border-radius:8px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🖐 Hand Gesture Recognition System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙ Controls")
start_button = st.sidebar.button("Start Camera")
stop_button = st.sidebar.button("Stop Camera")

st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Click Start Camera  
2. Place your hand in the green box  
3. Show gestures (0 - 5)  
4. Try OK (👌) and Best of Luck (👍)
""")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    gesture_placeholder = st.empty()
    mask_placeholder = st.empty()

# Session state
if "run" not in st.session_state:
    st.session_state.run = False

if start_button:
    st.session_state.run = True

if stop_button:
    st.session_state.run = False


def process_frame(frame):
    """Detect hand gesture from a video frame."""
    try:
        frame = cv2.flip(frame, 1)
        roi = frame[100:300, 100:300]

        # Draw ROI box
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Skin color range
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, np.ones((3, 3)), iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return frame, mask, "Put hand in the box"

        cnt = max(contours, key=cv2.contourArea)

        # Areas
        hull = cv2.convexHull(cnt)
        area_cnt = cv2.contourArea(cnt)
        area_hull = cv2.contourArea(hull)
        if area_cnt == 0:
            return frame, mask, "No hand detected"

        arearatio = ((area_hull - area_cnt) / area_cnt) * 100

        # Find defects
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if len(hull_idx) < 3:
            return frame, mask, "Move hand"

        defects = cv2.convexityDefects(cnt, hull_idx)

        count_defects = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, depth = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                a = math.hypot(end[0] - start[0], end[1] - start[1])
                b = math.hypot(far[0] - start[0], far[1] - start[1])
                c = math.hypot(end[0] - far[0], end[1] - far[1])

                # avoid divide by 0
                if b == 0 or c == 0:
                    continue

                cos_value = (b*b + c*c - a*a) / (2 * b * c)
                cos_value = max(-1, min(1, cos_value))
                angle = math.degrees(math.acos(cos_value))

                if angle <= 90 and depth > 10000:
                    count_defects += 1
                    cv2.circle(roi, far, 4, (255, 0, 0), -1)
                    cv2.line(roi, start, end, (0, 255, 0), 2)

        fingers = count_defects + 1

        # Gesture classification
        if fingers == 1:
            if area_cnt < 2000:
                gesture = "Put hand properly"
            else:
                if arearatio < 12:
                    gesture = "0 - Fist"
                elif arearatio < 17.5:
                    gesture = "👍 Best of Luck"
                else:
                    gesture = "1 - One Finger"

        elif fingers == 2:
            gesture = "2 - Two Fingers"

        elif fingers == 3:
            if arearatio < 27:
                gesture = "3 - Three Fingers"
            else:
                gesture = "👌 OK"

        elif fingers == 4:
            gesture = "4 - Four Fingers"

        elif fingers == 5:
            gesture = "5 - Five Fingers"

        else:
            gesture = "Reposition hand"

        return frame, mask, gesture

    except Exception as e:
        return frame, None, f"Error: {e}"


# Camera Loop
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Camera not detected.")
        st.session_state.run = False
    else:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error.")
                break

            processed_frame, mask, gesture = process_frame(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(processed_frame, channels="RGB")
            gesture_placeholder.markdown(f'<div class="gesture-display">{gesture}</div>', unsafe_allow_html=True)

            if mask is not None:
                mask_placeholder.image(mask, channels="GRAY")

        cap.release()
else:
    st.info("Click Start Camera to begin")
