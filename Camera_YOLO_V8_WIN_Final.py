import cv2
import datetime
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

class Camera:
    def __init__(self, mirror = False):
        self.cam = cv2.VideoCapture(0)
        self.now_recording = False
        self.success, self.data = self.cam.read()

        self.WIDTH = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2

    def show(self):
        while self.success:
            self.success, self.data = self.cam.read()
            frame = self.data
            if frame is not None:
                results = model.track(frame, persist=True, imgsz=self.WIDTH)
                pub_data = [results[0].boxes.cls, results[0].boxes.conf,
                            results[0].boxes.id, results[0].boxes.xywh, 
                            results[0].boxes.xywhn, results[0].boxes.xyxy,
                            results[0].boxes.xyxyn]
                # print(pub_data)

                annotated_frame = results[0].plot()
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
            if self.now_recording:
                out.write(frame)

            key = cv2.waitKey(1)
            if key == ord('r') and self.now_recording == False:
                self.now_recording = True
                now = datetime.datetime.now()
                filename =  now.strftime("%Y%m%d_%H%M%S") + ".avi"
                fps = 30
                out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, (self.WIDTH, self.HEIGHT))
                out.write(frame)
            elif key == ord('r') and self.now_recording == True:
                self.now_recording = False
                out.release()
                print(f"Recorded file saved: {filename}")
            elif key == ord('q'):
                # q : close
                self.release()
                if self.now_recording == True:
                    out.release()
                    print(f"Recorded file saved: {filename}")
                cv2.destroyAllWindows()
                break
            
    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cam = Camera()
    cam.show()

# https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes
# https://docs.ultralytics.com/reference/engine/results/?h=#ultralytics.engine.results.Boxes