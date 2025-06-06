import os
import sys
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# ✅ Add your project root to sys.path
sys.path.append('/content/drive/MyDrive/project')  # 🔁 change this based on your path

# ✅ Import after appending path
import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Initialize MTCNN face detector
detector = MTCNN()

# Define 16 random 5x5 filters (placeholders for texture features)
KERNELS = [np.random.rand(5, 5).astype(np.float32) for _ in range(16)]

def extract_texture_energy(region):
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = cv2.resize(region, (64, 64))
    return [np.mean(np.abs(cv2.filter2D(region, -1, k))) for k in KERNELS]

def extract_vaf(video_folder):
    eye_features, mouth_features = [], []
    frames = sorted(f for f in os.listdir(video_folder) if f.endswith(".jpg"))

    for frame in frames:
        img = cv2.imread(os.path.join(video_folder, frame))
        if img is None: continue
        results = detector.detect_faces(img)
        if not results: continue

        kp = results[0]["keypoints"]
        # Crop eye and mouth regions
        ex1, ex2 = min(kp["left_eye"][0], kp["right_eye"][0]) - 10, max(kp["left_eye"][0], kp["right_eye"][0]) + 10
        ey1, ey2 = min(kp["left_eye"][1], kp["right_eye"][1]) - 10, max(kp["left_eye"][1], kp["right_eye"][1]) + 10
        mx1, mx2 = min(kp["mouth_left"][0], kp["mouth_right"][0]) - 10, max(kp["mouth_left"][0], kp["mouth_right"][0]) + 10
        my1, my2 = min(kp["mouth_left"][1], kp["mouth_right"][1]) - 10, max(kp["mouth_left"][1], kp["mouth_right"][1]) + 10

        eye = img[max(ey1, 0):ey2, max(ex1, 0):ex2]
        mouth = img[max(my1, 0):my2, max(mx1, 0):mx2]

        if eye.size > 0: eye_features.append(extract_texture_energy(eye))
        if mouth.size > 0: mouth_features.append(extract_texture_energy(mouth))

    if not eye_features or not mouth_features:
        return np.zeros(32)

    return np.concatenate([np.mean(eye_features, axis=0), np.mean(mouth_features, axis=0)])


def generate_pseudo_labels(input_root, output_json="pseudo_labels1.json", n_clusters=2):
    features = []
    video_ids = []

    print(f"\n[INFO] Scanning video folders in: {input_root}")
    for vid_name in tqdm(os.listdir(input_root), desc="Extracting VAFs"):
        vid_folder = os.path.join(input_root, vid_name)
        if os.path.isdir(vid_folder):
            try:
                vaf = extract_vaf(vid_folder)
                features.append(vaf)
                video_ids.append(vid_name)
            except Exception as e:
                print(f"[WARNING] Failed to process {vid_name}: {e}")

    if not features:
        print("[ERROR] No valid features extracted. Exiting.")
        return

    print("[INFO] Clustering VAFs using KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    labels = {video_ids[i]: int(kmeans.labels_[i]) for i in range(len(video_ids))}

    print(f"[INFO] Saving pseudo-labels1 to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(labels, f, indent=4)

    print("[DONE] Pseudo-label generation complete.\n")

# ✅ Run
input_path = "/content/drive/MyDrive/dataset/preprocessd"
output_file = "/content/drive/MyDrive/dataset/pseudo_labels1.json"
generate_pseudo_labels(input_root=input_path, output_json=output_file)
