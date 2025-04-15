import cv2
import numpy as np
from ultralytics import YOLO
from tracking.custom_tracker import Tracker 
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from sklearn.cluster import DBSCAN

# === Setup ===
video_files = [
    "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_52.mp4",
    "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_139.mp4",
    "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_140.mp4",
    "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_142.mp4"
]

homography_files = [
    "/Users/adirajuadityasrivatsa/homographies/h_cam1.npy",
    "/Users/adirajuadityasrivatsa/homographies/h_cam2.npy",
    "/Users/adirajuadityasrivatsa/homographies/h_cam3.npy",
    "/Users/adirajuadityasrivatsa/homographies/h_cam4.npy"
]

model = YOLO("yolov8n.pt")
homographies = [np.load(h) for h in homography_files]
caps = [cv2.VideoCapture(v) for v in video_files]
tracker = Tracker(image_boundary=(1280, 800))  # match canvas size
colors = [(255,0,0), (0,255,0), (0,0,255), (0,140,255), (255,0,255), (0,255,255)]

# === Video dimensions ===
frame_w, frame_h = 640, 360
canvas_size = (500, frame_w * 2, 3)  # Top-down map same width as 2 camera frames
def cluster_topdown_bboxes(bboxes, eps=25, min_samples=1):
    # Convert bboxes to center points
    centers = np.array([
        [(x1 + x2)/2, (y1 + y2)/2]
        for (x1, y1, x2, y2) in bboxes
    ])
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    clustered_bboxes = []
    
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:
            continue  # noise
        members = np.where(clustering.labels_ == cluster_id)[0]
        cluster_boxes = bboxes[members]
        # Take the average of the boxes or just the first one
        avg_box = np.mean(cluster_boxes, axis=0)
        clustered_bboxes.append(avg_box)
    
    return np.array(clustered_bboxes)
# Optional video save
out = cv2.VideoWriter("dashboard_output_1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_w*2, frame_h*2 + canvas_size[0]))

while True:
    all_tracks = []
    camera_views = []
    detections = []

    for cam_id, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            camera_views.append(frame)
            continue

        frame = cv2.resize(frame, (frame_w, frame_h))
        results = model(frame)[0]

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_bottom = np.array([[[ (x1 + x2)//2, y2 ]]], dtype='float32')
            topdown_pos = cv2.perspectiveTransform(center_bottom, homographies[cam_id])
            x_mapped, y_mapped = map(int, topdown_pos[0][0])

            bbox_topdown = np.array([x_mapped-10, y_mapped-10, x_mapped+10, y_mapped+10])
            detections.append(bbox_topdown)

            # Draw detection box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        camera_views.append(frame)

    if detections:
        candidate_bboxes = cluster_topdown_bboxes(np.array(detections))
        tracker.update_tracker(candidate_bboxes, time_stamp=cv2.getTickCount())

    # === Build top-down canvas ===
    canvas = np.ones(canvas_size, dtype=np.uint8) * 255
    # Top-down room map size settings
    cv2.rectangle(canvas, (100, 100), (1100, 300), (220,220,220), 2)   # Room border
     # Object

    for traj in tracker.trajectories:
        if not traj.is_alive:
            continue
        x, y = traj.get_leatest_node_center()
        person_id = traj.object_id[-1]
        color = colors[person_id % len(colors)]
        cv2.circle(canvas, (x, y), 10, color, -1)
        cv2.putText(canvas, f"Person {person_id}", (x+12, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # === Compose dashboard ===
    row1 = cv2.hconcat(camera_views[0:2])
    row2 = cv2.hconcat(camera_views[2:4])
    dashboard = cv2.vconcat([row1, row2, canvas])

    cv2.imshow("Multi-Cam Dashboard", dashboard)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
for cap in caps:
    cap.release()
# out.release()
cv2.destroyAllWindows()