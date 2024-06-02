import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class TrackingProcessor(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    tracking_progress_updated = pyqtSignal(int, int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = cv2.TrackerMIL_create()
        self.running = True
        self.init_bounding_box = None
        self.orb = cv2.ORB_create()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.processed_frames = 0

    def set_bounding_box(self, bbox):
        self.init_bounding_box = bbox

    def run(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return

            if self.init_bounding_box is None:
                return

            self.tracker.init(frame, self.init_bounding_box)
            self.kp_init, self.des_init = self.orb.detectAndCompute(frame, None)

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.processed_frames += 1
                self.tracking_progress_updated.emit(self.processed_frames, self.total_frames)

                ret, bbox = self.tracker.update(frame)
                if not ret:
                    # Tracker lost the target, try to reinitialize using ORB feature matching
                    kp_frame, des_frame = self.orb.detectAndCompute(frame, None)
                    if des_frame is not None and len(des_frame) >= 2:
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(self.des_init, des_frame)
                        matches = sorted(matches, key=lambda x: x.distance)
                        if len(matches) > 10:
                            src_pts = np.float32([self.kp_init[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            h, w = frame.shape[:2]
                            pts = np.float32([[self.init_bounding_box[0], self.init_bounding_box[1]],
                                              [self.init_bounding_box[0] + self.init_bounding_box[2], self.init_bounding_box[1]],
                                              [self.init_bounding_box[0] + self.init_bounding_box[2], self.init_bounding_box[1] + self.init_bounding_box[3]],
                                              [self.init_bounding_box[0], self.init_bounding_box[1] + self.init_bounding_box[3]]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            self.init_bounding_box = cv2.boundingRect(dst)
                            self.tracker = cv2.TrackerMIL_create()
                            self.tracker.init(frame, self.init_bounding_box)

                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                # Convert the frame from BGR to RGB for correct color display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame)
        except Exception as e:
            print(f"Error in tracking process: {e}")
        finally:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()
