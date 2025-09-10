import streamlit as st
import cv2
import face_recognition
import numpy as np
import pickle
import os
from twilio.rest import Client

# -------------------- Configuration --------------------
TWILIO_ACCOUNT_SID = 'ACa502df151ae05f655d2795ed2f459a46'
TWILIO_AUTH_TOKEN = '682bb417c59e73922f3db30e1ff2328d'
TWILIO_PHONE_NUMBER = '++17752628578'
ALERT_PHONE_NUMBER = '+917569568080'  # Change to your actual number
SECURE_PASSCODE = 'admin123'  # Change to a secret

ENCODINGS_FILE = "known_faces.pkl"

# -------------------- Helper Functions --------------------
def load_known_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return [], []

def save_known_faces(encodings, names):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((encodings, names), f)

def send_sms_alert(name):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"⚠️ Alert: Unknown person detected!",
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        st.warning("SMS Alert Sent!")
    except Exception as e:
        st.error(f"Error sending SMS: {e}")

# -------------------- Face Registration --------------------
def register_face():
    st.subheader("🔐 Register New Face")
    passcode = st.text_input("Enter Passcode", type="password", key="passcode")

    if passcode == SECURE_PASSCODE:
        run = st.checkbox("Activate Camera", key="reg_camera")
        FRAME_WINDOW = st.image([])

        known_encodings, known_names = load_known_faces()

        camera = cv2.VideoCapture(0)

        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            face_locations = face_recognition.face_locations(frame_rgb)
            if face_locations:
                st.success("Face Detected!")
                name = st.text_input("Enter Name", key="reg_name")
                if st.button("Save Face"):
                    face_encoding = face_recognition.face_encodings(frame_rgb, face_locations)[0]
                    known_encodings.append(face_encoding)
                    known_names.append(name)
                    save_known_faces(known_encodings, known_names)
                    st.success(f"Face for '{name}' registered!")
                    break
        camera.release()
    elif passcode:
        st.error("Invalid passcode.")

# -------------------- Live Surveillance --------------------
def live_surveillance():
    st.subheader("📹 Live Surveillance")
    run = st.checkbox("Start Live Feed", key="live_camera")
    FRAME_WINDOW = st.image([])

    known_encodings, known_names = load_known_faces()
    camera = cv2.VideoCapture(0)
    unknown_detected = False

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                else:
                    if not unknown_detected:
                        send_sms_alert(name)
                        unknown_detected = True

            face_names.append(name)

        # Display names on frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Smart CCTV Face Recognition", layout="centered")
st.title("🔍 Smart CCTV Face Recognition")
st.markdown("Built with Streamlit, OpenCV, Face Recognition, and Twilio")

mode = st.sidebar.selectbox("Choose Mode", ["Live Surveillance", "Register New Face"])

if mode == "Live Surveillance":
    live_surveillance()
elif mode == "Register New Face":
    register_face()
