
from ultralytics import YOLO
import cv2
import os

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("input/cctv.mp4")

if not video.isOpened():
    print("Error: Could not open video")
    exit()

os.makedirs("output", exist_ok=True)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

output = cv2.VideoWriter(
    "output/tracked_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

count = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    count += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        conf=0.35,
        verbose=False
    )

    for result in results:

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = "Vehicle ID: " + str(int(track_id))

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    output.write(frame)

    cv2.imshow("Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
output.release()
cv2.destroyAllWindows()

print("Vehicle tracking completed!")
print("Processed frames:", count)
print("Saved video: output/tracked_video.mp4")
