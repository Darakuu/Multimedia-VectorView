import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class TrackingProcessor(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    tracking_progress_updated = pyqtSignal(int, int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.tracker = cv2.TrackerMIL_create()
        self.is_running = True
        self.initial_bbox = None
        self.orb_detector = cv2.ORB_create()
        self.total_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_count = 0

    def set_bounding_box(self, bbox):
        self.initial_bbox = bbox

    def run(self):
        try:
            frame_read, frame = self.video_capture.read()
            if not frame_read or self.initial_bbox is None:
                return

            # Initialize the tracker with the first frame and the initial bounding box
            self.tracker.init(frame, self.initial_bbox)
            self.keypoints_initial, self.descriptors_initial = self.orb_detector.detectAndCompute(frame, None)

            while self.is_running:
                frame_read, frame = self.video_capture.read()
                if not frame_read:
                    break

                self.current_frame_count += 1
                self.tracking_progress_updated.emit(self.current_frame_count, self.total_frame_count)

                frame_read, bbox = self.tracker.update(frame)
                if not frame_read:
                    # Tracker lost the target, try to reinitialize using ORB feature matching
                    keypoints_frame, descriptors_frame = self.orb_detector.detectAndCompute(frame, None)
                    # Check if there are enough keypoints in the current frame
                    if descriptors_frame is not None and len(descriptors_frame) >= 2:
                        # Create a BFMatcher object using Hamming distance
                        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        # Match the initial descriptors with the current frame descriptors
                        matches = bf_matcher.match(self.descriptors_initial, descriptors_frame)
                        # Sort matches by distance
                        matches = sorted(matches, key=lambda x: x.distance)
                        if len(matches) > 10:
                            src_points = np.float32(
                                [self.keypoints_initial[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
                            dst_points = np.float32([keypoints_frame[m.trainIdx].pt for m in matches[:10]]).reshape(-1,
                                                                                                                    1,
                                                                                                                    2)

                            # Uses: RANdom SAmple Consensus, findHomography from Features2D:
                            # https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
                            # Find the homography matrix to transform the source points to destination points
                            matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

                            # Define the four corners of the initial bounding box
                            bbox_points = np.float32([
                                [self.initial_bbox[0], self.initial_bbox[1]],  # Top-left corner
                                [self.initial_bbox[0] + self.initial_bbox[2], self.initial_bbox[1]],  # Top-right corner
                                [self.initial_bbox[0] + self.initial_bbox[2],
                                 self.initial_bbox[1] + self.initial_bbox[3]],  # Bottom-right corner
                                [self.initial_bbox[0], self.initial_bbox[1] + self.initial_bbox[3]]
                                # Bottom-left corner
                            ]).reshape(-1, 1, 2)  # Reshape to the required format for perspective transform
                            transformed_bbox = cv2.perspectiveTransform(bbox_points, matrix)
                            self.initial_bbox = cv2.boundingRect(transformed_bbox)
                            self.tracker = cv2.TrackerMIL_create()
                            self.tracker.init(frame, self.initial_bbox)

                top_left = (int(bbox[0]), int(bbox[1]))
                bottom_right = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2, 1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)
        except Exception as e:
            print(f"Error in tracking process: {e}")
        finally:
            self.video_capture.release()

    def stop(self):
        self.is_running = False
        self.wait()
