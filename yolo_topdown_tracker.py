import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# === Setup ===
video_files = [
    "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_52.mp4",
    "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_139.mp4",
    "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_140.mp4",
    "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_142.mp4"
]

homography_files = [
    "homographies/h_cam1.npy",
    "homographies/h_cam2.npy",
    "homographies/h_cam3.npy",
    "homographies/h_cam4.npy"
]

model = YOLO("yolov8s.pt")
homographies = [np.load(h) for h in homography_files]
caps = [cv2.VideoCapture(v) for v in video_files]
global_tracker = [DeepSort(max_age=30) for _ in video_files]
colors = [(255,0,0), (0,255,0), (0,0,255), (0,140,255), (255,0,255), (0,255,255)]

# === Video dimensions ===
frame_w, frame_h = 640, 360
canvas_size = (800, frame_w * 2, 3)  # Top-down map same width as 2 camera frames


# out = cv2.VideoWriter("dashboard_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_w*2, frame_h*2 + canvas_size[0]))

while True:
    all_tracks = []
    camera_views = []

    for cam_id, (cap, tracker) in enumerate(zip(caps, global_tracker)):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            camera_views.append(frame)
            continue

        frame = cv2.resize(frame, (frame_w, frame_h))
        results = model(frame)[0]
        detections = []
        conf_threshold=0.45
        min_width=30
        min_height=50
        min_aspect=1.1
        max_aspect=4.5
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue  # not a person

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / (width + 1e-5)

            # Debug
            print(f"[YOLO] Person: conf={conf:.2f}, w={width}, h={height}, AR={aspect_ratio:.2f}")

            # Adaptive filtering
            if conf < 0.4:
                print("❌ Rejected: Low confidence")
                continue
            if width < 20 or height < 35:
                print("❌ Rejected: Too small")
                continue
            if not (0.9 <= aspect_ratio <= 5.0):
                print("❌ Rejected: Bad aspect ratio")
                continue

            print("✅ Passed filter")
            detections.append(([x1, y1, width, height], conf, 'person'))

            # Optional: draw box directly
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"YOLO {conf:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        tracks = global_tracker.update_tracks(detections, frame=frame)
        # === DeepSort tracking ===

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # Use the center of the bounding box as the projection point
            x_proj = (x1 + x2) // 2
            y_proj = int(0.8 * ((y1 + y2) // 2) + 0.2 * y2)

            project_point = np.array([[[x_proj, y_proj]]], dtype='float32')
            topdown_pos = cv2.perspectiveTransform(project_point, homographies[cam_id])
            scale=0.5
            x_mapped, y_mapped = map(int, topdown_pos[0][0]*scale)
            print(f"[TopDown] Track {track_id} from cam {cam_id} => ({x_mapped}, {y_mapped})")

            # Only add if within bounds
            canvas_w = frame_w * 2
            canvas_h = 400
            if 0 <= x_mapped < canvas_w and 0 <= y_mapped < canvas_h:
                all_tracks.append((track_id, x_mapped, y_mapped))
            else:
                print(f"⚠️ Track {track_id} mapped out of bounds, skipping: ({x_mapped}, {y_mapped})")

            # Draw on frame
            color = colors[track_id % len(colors)]
            label = f"Person {track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Append final frame (after all tracks drawn)
        camera_views.append(frame)

    # === Build top-down canvas ===
    canvas = np.ones(canvas_size, dtype=np.uint8) * 255
    cv2.rectangle(canvas, (100, 100), (1100, 300), (220,220,220), 2)  # Room border
    cv2.rectangle(canvas, (300, 150), (500, 250), (150, 150, 255), -1)  # Object

    for track_id, x, y in all_tracks:
        color = colors[track_id % len(colors)]
        cv2.circle(canvas, (x, y), 10, color, -1)
        cv2.putText(canvas, f"Person {track_id}", (x+12, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # === Compose dashboard ===
    row1 = cv2.hconcat(camera_views[0:2])
    row2 = cv2.hconcat(camera_views[2:4])
    dashboard = cv2.vconcat([row1, row2, canvas])

    cv2.imshow("Multi-Cam Dashboard", dashboard)
    # out.write(dashboard)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for cap in caps:
    cap.release()
# out.release()
cv2.destroyAllWindows()