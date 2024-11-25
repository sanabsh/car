from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from android.storage import app_storage_path
import cv2
from ultralytics import YOLO
from flask import Flask, Response, request
import threading
import socket
import usb4a  # For Arduino communication on Android

# Initialize Flask app for streaming and control
flask_app = Flask(__name__)

# Adjust path for YOLO model
model_path = app_storage_path() + '/yolov8n.pt'

# Arduino setup
arduino = None
try:
    arduino = usb4a.Serial('/dev/ttyUSB0', 9600)
except Exception as e:
    print("Arduino not connected:", e)

class VideoStreamApp(App):
    def build(self):
        # Initialize YOLO model for filtering detections
        self.model = YOLO(model_path)
        self.camera = cv2.VideoCapture(0)

        # Create layout to hold the video stream and IP address label
        layout = BoxLayout(orientation='vertical')

        # Display IP address label
        self.ip_label = Label(text="Stream IP: " + self.get_local_ip(), size_hint_y=None, height=50)
        layout.add_widget(self.ip_label)

        # Create an Image widget to display video frames
        self.img_widget = Image()
        layout.add_widget(self.img_widget)

        # Schedule video stream update for Kivy display
        self.frame_event = Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 30 FPS

        # Start Flask server in a separate daemon thread
        self.flask_thread = threading.Thread(target=flask_app.run, kwargs={'host': '0.0.0.0', 'port': 5000})
        self.flask_thread.daemon = True  # Set as daemon so it closes with the main app
        self.flask_thread.start()

        return layout

    def get_local_ip(self):
        # Retrieve the car’s phone’s local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = 'IP not found'
        finally:
            s.close()
        return ip

    def update_frame(self, dt):
        # Capture frame-by-frame from the camera
        success, frame = self.camera.read()
        if not success:
            return

        # Run YOLO object detection on the frame
        results = self.model(frame, stream=True)

        # Process detections and filter for "person" and "vehicle" classes
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                class_name = self.model.names[int(cls)]
                if class_name in ["person", "car", "bus", "truck", "bicycle", "motorbike"]:
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to Kivy texture and display
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img_widget.texture = texture
        self.current_frame = frame

    def on_stop(self):
        # Stop the scheduled frame updates
        if self.frame_event:
            self.frame_event.cancel()

        # Release the video capture on app close
        self.camera.release()

        # Close Arduino connection if available
        if arduino:
            arduino.close()

# Flask route for video streaming
@flask_app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            success, jpeg = cv2.imencode('.jpg', App.get_running_app().current_frame)
            if not success:
                continue
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to receive control commands
@flask_app.route('/control', methods=['GET'])
def control():
    direction = request.args.get('direction')
    if not direction:
        return "No direction provided", 400  # Bad request if direction is missing

    # Simulate sending a command to either a real or mock Arduino
    command = direction[0].upper()  # Extract first letter of direction ('F', 'B', 'L', 'R')

    if arduino:  # If a real Arduino is connected
        try:
            arduino.write(command.encode())  # Send command to Arduino
            print(f"Sent command to Arduino: {command}")
            return "Command sent", 200
        except Exception as e:
            print(f"Error sending command to Arduino: {e}")
            return "Error communicating with Arduino", 500
    else:
        # For testing with mock Arduino, log the command
        print(f"Simulating command '{command}' for direction '{direction}' with mock Arduino.")
        return f"Mock command '{command}' received", 200

if __name__ == '__main__':
    VideoStreamApp().run()
