from flask import Flask, request, jsonify
import requests
import tempfile
import time
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mutils import convert
import stats_model_testing as stats
import os
import svm_testing as svm

# Load the model from local directory
MODEL_NAME = "trained_with_20_files"
model = tf.keras.models.load_model(f"./model/{MODEL_NAME}/model.h5")

# Define MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Flask app
app = Flask(__name__)
app.debug = True


@app.route('/', methods=['POST'])
def predict():
    start_time = time.time()
    print("start time:0.0 seconds")

    url = request.get_json()['url']
    filename = download_file(url)

    download_time = time.time()
    download_time = download_time - start_time
    print(f"finished download video: {download_time:.4f} seconds")

    try:
        # Process video
        result = process_video(filename, start_time)
        statuses = list(result.values())

        # Convert the list of statuses to a string
        statuses_str = ', '.join(statuses)
        os.remove(filename)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    end_time = time.time()
    finish_everything = end_time - start_time
    print(f"finished everything: {finish_everything:.4f} seconds")

    return jsonify(statuses_str)


@app.route("/verify", methods=['POST'])
def verify():
    print("ENDPOINT WAS CALLLEDDDD\n")
    if "photo" not in request.files:
        return jsonify("Error: No photo part of request", 400)

    static_pose = mp_pose.Pose(min_detection_confidence=0.5, static_image_mode=True)
   
    photo_file = request.files['photo']
    photo_file_bytes = np.frombuffer(photo_file.read(), np.uint8)
    image = cv2.imdecode(photo_file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "invalid image file"}), 400

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = static_pose.process(image)

    if (results.pose_landmarks):
        print("Patient body identified")
        return jsonify({"status": "success", "message": "Patient body identified"}), 200
    else:
        print("Patient body not detected")
        return jsonify({"status": "fail", "message": "Patient body not identified"}), 200


@app.route("/skibidi", methods=['POST'])
def test():
    print("yippee")
    if "photo" not in request.files:
        return jsonify("Error: No photo part of request", 400)
    
    

def download_file(url):
    r = requests.get(url, stream=True)
    r.raise_for_status()  # Raise exception if invalid response
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")  # Create a temp file with .mp4 extension
    with open(temp.name, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return temp.name


def process_video(video_file, start_time):
    cap = cv2.VideoCapture(video_file)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_data = []  # This will store the 2D array with 3-length sub-arrays

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break  # Exit loop if no more frames

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            if results.pose_landmarks is not None:
                # Assuming convert function flattens landmarks to a list of 27 elements
                converted_landmarks = convert(results.pose_world_landmarks.landmark)

                chunk = []
                for i in range(0, len(converted_landmarks), 3):
                    chunk.append(converted_landmarks[i])
                    chunk.append(converted_landmarks[i + 1])

                    if len(chunk) == 18:
                        frame_data.append(chunk)

        timer_for_svm = time.time()
        # Now frame_data is prepared as required: a 2D array with sub-arrays of length 3
        res = svm.test_svm(frame_data, "./svm.joblib", "./svm_parameters.txt")

        finish_svm = time.time()
        finish_svm = finish_svm - timer_for_svm
        print(f"finished svm: {finish_svm:.4f} seconds")


    except Exception as e:
        app.logger.error(f"Error in process_video: {str(e)}")
        return {"error": str(e)}
    finally:
        cap.release()

    return res

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)