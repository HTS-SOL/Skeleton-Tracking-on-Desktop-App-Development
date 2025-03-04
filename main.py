import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit Page Configuration
st.set_page_config(page_title="Skeleton Tracking App", layout="wide")

# Sidebar Menu
st.sidebar.title("📌 Skeleton Tracking Options")
option = st.sidebar.radio("Choose an option:", ["Live Webcam", "Process Video File"])

# Temporary file storage for video processing
temp_video_file = None

# Function to process video file
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default FPS if not detected
    
    # Create a temporary file for saving output video
    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    
    pose = mp_pose.Pose()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()
    
    return temp_output_path  # Return the processed video file path

# Function for live webcam tracking (opens in popup)
def live_webcam():
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Live Skeleton Tracking (Press Q to Exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Handle user choice
if option == "Live Webcam":
    st.sidebar.warning("⚠️ Live video will open in a separate popup window.")
    if st.sidebar.button("Start Live Tracking"):
        live_webcam()

elif option == "Process Video File":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        # Save the uploaded file to a temporary location
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_file, "wb") as f:
            f.write(uploaded_file.read())

        st.sidebar.success("✅ Video uploaded successfully!")
        
        if st.sidebar.button("Start Processing"):
            with st.spinner("⏳ Processing video... Please wait."):
                processed_video_path = process_video(temp_video_file)
            
            st.success("🎉 Video processing complete!")
            st.video(processed_video_path)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ using Streamlit, OpenCV, and MediaPipe")
