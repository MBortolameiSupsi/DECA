import cv2
import mediapipe as mp
import argparse
import os
import numpy as np
import open3d as o3d

def detect_face_bbox(image, face_mesh):
    # model = mp_face_mesh.FaceMesh()

    h,w = image.shape[:2]
    # out = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    out = face_mesh.process(image[:,:,::-1])
    if out.multi_face_landmarks:
        face_landmarks = out.multi_face_landmarks[0]
        pts = np.array([(pt.x * w, pt.y * h)
                        for pt in face_landmarks.landmark],
                       dtype=np.float64)
        min_coords = pts.min(axis=0)
        max_coords = pts.max(axis=0)
        bbox = [
            int(min_coords[0]),  # left
            int(min_coords[1]),  # top
            int(max_coords[0]),  # right
            int(max_coords[1])   # bottom
        ]
        mediapipe_landmarks = np.asarray([np.array([landmark.x, landmark.y, landmark.z]) for landmark in face_landmarks.landmark])
        
        # bbox = np.round(bbox).astype(np.int32)
        return bbox, mediapipe_landmarks
    return None

def detect_face_bbox_original(image, face_mesh):
    """Detect face and return bounding box using MediaPipe FaceMesh."""
    # Convert the BGR image to RGB.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image.
    result = face_mesh.process(rgb_image)
    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        x_coords = [lm.x for lm in face_landmarks.landmark]
        y_coords = [lm.y for lm in face_landmarks.landmark]
        # Convert normalized coordinates to pixel coordinates.
        width, height = image.shape[1], image.shape[0]
        x_coords = np.array(x_coords) * width
        y_coords = np.array(y_coords) * height
        # Determine the bounding box of the face.
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        return bbox
    return None

def main(input_path, output_path, use_webcam):
    """Process images from a folder or webcam."""
    if use_webcam:
        # Start webcam capture.
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
    else:
        # Get list of image files.
        files = [f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg'))]

    # Initialize MediaPipe FaceMesh.
    mp_face_mesh = mp.solutions.face_mesh
    # face_mesh = mp_face_mesh.FaceMesh(static_image_mode=not use_webcam, max_num_faces=1)
    face_mesh = mp_face_mesh.FaceMesh()

    frame_count = 0
    bbox_file = open(os.path.join(output_path, 'bboxes.txt'), 'w')

    while True:
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_count >= len(files):
                break
            frame = cv2.imread(os.path.join(input_path, files[frame_count]))

        # Detect face and draw bounding box.
        bbox, mediapipe_landmarks = detect_face_bbox(frame, face_mesh)
        if bbox:
            # left, top, right, bottom = map(int, bbox)
            left, top, right, bottom = bbox
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            bbox_file.write(f"{frame_count}: {bbox}\n")
        # else:
        #     bbox_file.write(f'Frame {frame_index}: No face detected\n')
        # Save or display the processed frame.
        if use_webcam:
            cv2.imshow('Webcam Feed', frame)
            # Save original and processed frame.
            cv2.imwrite(os.path.join(output_path, f'frame_{frame_count}_original.jpg'), frame)
            cv2.imwrite(os.path.join(output_path, f'frame_{frame_count}_processed.jpg'), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.imwrite(os.path.join(output_path, f'frame_{frame_count}_processed.jpg'), frame)
            mediapipe_landmarks_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mediapipe_landmarks))
            o3d.io.write_point_cloud(os.path.join(output_path, f'mediapipe_landmarks_{frame_count}.ply'), mediapipe_landmarks_cloud)
        frame_count += 1

    bbox_file.close()
    if use_webcam:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Bounding Box Detection with MediaPipe")
    parser.add_argument("--input", help="Input folder path for image files", default=None)
    parser.add_argument("--output", help="Output folder path to save processed images", required=True)
    parser.add_argument("--webcam", help="Use webcam as input source", action="store_true")
    args = parser.parse_args()

    if not args.webcam and not args.input:
        print("Please specify an input directory with --input or use --webcam for live capture.")
    else:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        main(args.input, args.output, args.webcam)