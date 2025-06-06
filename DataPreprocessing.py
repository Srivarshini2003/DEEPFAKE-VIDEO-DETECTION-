import os
import cv2
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
from datetime import datetime

# Initialize MTCNN for face detection
detector = MTCNN()

def extract_faces_from_video(video_path, output_dir, resize_dim=(299, 299), max_frames=32):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    saved = 0
    frame_index = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames at intervals
        if frame_index % interval == 0:
            results = detector.detect_faces(frame)
            print(f"Frame {frame_index}: {len(results)} faces detected.")
            if results:
                x, y, w, h = results[0]['box']
                x, y = max(x, 0), max(y, 0)
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, resize_dim)
                filename = os.path.join(output_dir, f"frame_{saved}.jpg")
                cv2.imwrite(filename, face)
                saved += 1

        frame_index += 1

    cap.release()
    print(f"Saved {saved} face frames from {video_path}")
    return saved

def process_videos(input_dir, output_dir, log_file="face_extraction_log.txt"):
    print(f"Checking input directory: {input_dir}")
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    videos = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
    print(f"Found {len(videos)} videos: {videos}")

    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise FileExistsError(f"Output path '{output_dir}' exists and is not a directory.")
    else:
        os.makedirs(output_dir, exist_ok=True)

    with open(log_file, 'a') as log:
        log.write(f"\n--- Extraction Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        for video in tqdm(videos, desc="Processing videos"):
            video_path = os.path.join(input_dir, video)
            vid_name = os.path.splitext(video)[0]
            save_dir = os.path.join(output_dir, vid_name)
            os.makedirs(save_dir, exist_ok=True)

            try:
                num_faces = extract_faces_from_video(video_path, save_dir)
                log.write(f"{video}: {num_faces} face frames saved in {save_dir}\n")
            except Exception as e:
                log.write(f"{video}: FAILED - {str(e)}\n")
                print(f"Error processing {video}: {e}")
        log.write("---------------------------------------------------------\n")

if __name__ == "__main__":
    # Update these paths as needed
    input_folder = "/content/drive/MyDrive/dataset/Fake videos"
    output_folder = "/content/Preprocessed_data"

    print("Starting video processing...")
    process_videos(input_folder, output_folder)
    print("Processing complete.")
