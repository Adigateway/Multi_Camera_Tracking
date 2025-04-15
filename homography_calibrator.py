import cv2
import numpy as np
import os

# Store image points clicked by user
image_points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        image_points.append([x, y])
        print(f"Clicked image point: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click 4 ground points", img)

# === Load frame from your video ===
video_path = "/Users/adirajuadityasrivatsa/Documents/M/Sentics_project/cam_139.mp4"  
cap = cv2.VideoCapture(video_path)

ret, img = cap.read()
if not ret:
    raise ValueError("Could not read frame from video")

cv2.imshow("Click 4 ground points", img)
cv2.setMouseCallback("Click 4 ground points", click_event)

print("Please click 4 GROUND points in clockwise or counter-clockwise order.")

cv2.waitKey(0)
cv2.destroyAllWindows()

# === Define corresponding real-world (top-down) coordinates ===


world_points = [
    [100, 100],  # Corresponds to image_points[0]
    [700, 100],  # image_points[1]
    [700, 700],  # image_points[2]
    [100, 700]   # image_points[3]
]

if len(image_points) != 4:
    raise ValueError("You must click exactly 4 points.")

# === Compute homography matrix ===
H, _ = cv2.findHomography(np.array(image_points), np.array(world_points))
print("Homography matrix:\n", H)

# === Save it ===
os.makedirs("homographies", exist_ok=True)
np.save("homographies/h_cam2.npy", H)
print("Saved homography matrix to 'homographies/h_cam2.npy'")