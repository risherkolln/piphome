from ultralytics import YOLO
import cv2
import cvzone
import math
from onvif import ONVIFCamera
from flask import Flask, Response
import threading
import time
import base64

app = Flask(__name__)


print("Init camera...")
cam = ONVIFCamera('192.168.0.185', 8899, 'admin', '')
print("Camera ok...")

# Create media service
media = cam.create_media_service()

# Get video sources
video_sources = media.GetVideoSources()

# Select the first video source (assuming only one source)
video_source = video_sources[0]

# Get the profiles
profiles = media.GetProfiles()

# Assuming you want to get the stream URI for the first profile
if profiles:
    profile_token = profiles[0].token
    stream_uri = media.GetStreamUri({'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}, 'ProfileToken': profile_token})
    print("Stream URI:", stream_uri)
else:
    print("No profiles found")

# OpenCV VideoCapture to access the stream
cap = cv2.VideoCapture(stream_uri.Uri)

cap.set(3, 800)
cap.set(4, 600)

model = YOLO('../YOLO Weights/yolov8n.pt')

classNames = ["person",
              "bicycle",
              "car",
              "motorbike",
              "aeroplane",
              "bus",
              "train",
              "truck",
              "boat",
              "traffic light",
              "fire hydrant",
              "stop sign",
              "parking meter",
              "bench",
              "bird",
              "cat",
              "dog",
              "horse",
              "sheep",
              "cow",
              "elephant",
              "bear",
              "zebra",
              "giraffe",
              "backpack",
              "umbrella",
              "handbag",
              "tie",
              "suitcase",
              "frisbee",
              "skis",
              "snowboard",
              "sports ball",
              "kite",
              "baseball bat",
              "baseball glove",
              "skateboard",
              "surfboard",
              "tennis racket",
              "bottle",
              "wine glass",
              "cup",
              "fork",
              "knife",
              "spoon",
              "bowl",
              "banana",
              "apple",
              "sandwich",
              "orange",
              "broccoli",
              "carrot",
              "hot dog",
              "pizza",
              "donut",
              "cake",
              "chair",
              "sofa",
              "pottedplant",
              "bed",
              "diningtable",
              "toilet",
              "tvmonitor",
              "laptop",
              "mouse",
              "remote",
              "keyboard",
              "cell phone",
              "microwave",
              "oven",
              "toaster",
              "sink",
              "refrigerator",
              "book",
              "clock",
              "vase",
              "scissors",
              "teddy bear",
              "hair drier",
              "toothbrush"]


global_image = None
success, global_image = cap.read()

def process_cam_image():
    global global_image
    while True:
        success, frame = cap.read()
        global_image = frame.copy()
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1
                cvzone.cornerRect(frame, (x1, y1, w, h))

                conf = math.ceil((box.conf[0]*100))/100

                cls = box.cls[0]

                name = classNames[int(cls)]

                cvzone.putTextRect(frame, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 1.5)

        #cv2.imshow("Image", frame)
        cv2.waitKey(1)

capture_thread = threading.Thread(target=process_cam_image)

# Start the thread
capture_thread.daemon = True
capture_thread.start()

@app.route('/cam')
def get_image():
    global global_image
    image_copy = global_image.copy()
    if image_copy is None:
        return "No image available", 404

    # Encode the current frame as JPEG
    _, buffer = cv2.imencode('.jpg', image_copy)
    image_bytes = buffer.tobytes()

    # Return the image data with appropriate content type
    return Response(image_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='192.168.0.105', debug=False)