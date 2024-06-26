﻿import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel,
    QVBoxLayout, QPushButton, QWidget, QHBoxLayout,
    QGroupBox, QTabWidget, QMessageBox, QProgressBar, QLineEdit, QFormLayout, QScrollArea, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from source.ebma import ebma_search
from source.threestepsearch import tss_search
from ROITracking import TrackingProcessor


# Todo: Could be moved to its own file
class VideoProcessor(QThread):
    # Signals
    frame_ready = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int, int)

    def __init__(self, video_path, algorithm, block_size=16, search_radius=8, similarity_metric="MAD"):
        super().__init__()
        self.video_path = video_path  # Path to the video file
        self.block_size = block_size  # Block size for motion estimation algorithms
        self.search_radius = search_radius  # Search area for motion estimation algorithms
        self.videocapture = cv2.VideoCapture(video_path)  # Open the video file
        self.prev_frame = None  # Previous frame for motion estimation
        self.running = True  # Flag to stop the thread
        self.algorithm = algorithm  # The motion estimation algorithm to use
        self.similarity_metric = similarity_metric  # The similarity metric to use (MAD or SSD)
        self.total_frames = int(
            self.videocapture.get(cv2.CAP_PROP_FRAME_COUNT))  # Total Amount of frames in the video, needed for progress
        self.current_frame_index = 0  # Current frame index for resuming playback

    def run(self):
        if self.current_frame_index > 0:
            self.videocapture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

        while self.running:
            frame_read, frame = self.videocapture.read()
            if not frame_read:
                break

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.prev_frame is not None:
                    motion_vectors = self.calculate_motion_vectors(self.prev_frame, gray)
                    frame = self.draw_motion_vectors(frame, motion_vectors)
                self.prev_frame = gray

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame)

                # update progress bar
                self.current_frame_index += 1
                self.progress_updated.emit(self.current_frame_index, self.total_frames)

            except Exception as e:
                print(f"Error processing frame: {e}")
                break

        self.videocapture.release()

    def calculate_motion_vectors(self, prev_frame, curr_frame):
        try:
            return self.algorithm(prev_frame, curr_frame, self.block_size, self.search_radius, self.similarity_metric)
        except Exception as e:
            print(f"Error in calculating motion vectors: {e}")
            return []

    def draw_motion_vectors(self, frame: np.ndarray, motion_vectors: np.ndarray):
        """
        Draw motion vectors on the frame.

        Input:
            frame (np.ndarray): The frame on which to draw the motion vectors. It should be a 3-channel image.
            motion_vectors (np.ndarray): A 2D array of motion vectors. Each element is a tuple (dy, dx) representing
                                         the displacement vector for each block.

        Returns:
            np.ndarray: The frame with motion vectors drawn on it.
        """
        try:
            num_blocks_y, num_blocks_x, _ = motion_vectors.shape
            for block_y in range(num_blocks_y):
                for block_x in range(num_blocks_x):
                    dy, dx = motion_vectors[block_y, block_x]
                    start_point = (block_x * self.block_size + self.block_size // 2,
                                   block_y * self.block_size + self.block_size // 2)
                    if dx == 0 and dy == 0:  # Check if there was any change from the previous frame
                        end_point = start_point
                    else:
                        end_point = (int(start_point[0] + dx), int(start_point[1] + dy))
                    frame = cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 1)
            return frame
        except Exception as e:
            print(f"Error drawing motion vectors: {e}")
            return frame

    def stop(self):
        self.running = False
        self.wait()
        if self.videocapture.isOpened():
            self.current_frame_index = int(self.videocapture.get(cv2.CAP_PROP_POS_FRAMES))
            self.videocapture.release()

    def resume(self):
        if not self.videocapture.isOpened():
            self.videocapture = cv2.VideoCapture(self.video_path)
            self.running = True
            self.start()


class MotionVectorVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video_processor = None
        self.tracking_processor = None
        self.algorithm = None
        self.similarity_metric = "MAD"
        self.video_path = None
        self.bounding_box = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_frame = None
        self.tracking_started = False

    def initUI(self):
        self.setWindowTitle("Multimedia VectorView")
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.motion_tab = QWidget()
        self.motion_layout = QHBoxLayout(self.motion_tab)
        self.tabs.addTab(self.motion_tab, "Motion Estimation")

        self.video_layout = QVBoxLayout()
        self.motion_layout.addLayout(self.video_layout, stretch=4)

        self.video_label = QLabel()
        self.video_label.setScaledContents(True)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.video_label)
        self.video_layout.addWidget(self.scroll_area)

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.video_layout.addWidget(self.load_button)

        self.stop_button = QPushButton("Stop Video")
        self.stop_button.clicked.connect(self.stop_video)
        self.video_layout.addWidget(self.stop_button)

        self.resume_button = QPushButton("Resume Video")
        self.resume_button.clicked.connect(self.resume_video)
        self.video_layout.addWidget(self.resume_button)

        self.progress_bar = QProgressBar()
        self.video_layout.addWidget(self.progress_bar)

        self.side_menu_layout = QVBoxLayout()
        self.motion_layout.addLayout(self.side_menu_layout, stretch=1)

        self.algorithm_group_box = QGroupBox("Motion Estimation Algorithms")
        self.algorithm_layout = QVBoxLayout()
        self.algorithm_group_box.setLayout(self.algorithm_layout)
        self.side_menu_layout.addWidget(self.algorithm_group_box)

        self.ebma_button = QPushButton("EBMA")
        self.ebma_button.clicked.connect(lambda: self.set_algorithm(ebma_search))
        self.algorithm_layout.addWidget(self.ebma_button)

        self.tss_button = QPushButton("Three-Step-Search")
        self.tss_button.clicked.connect(lambda: self.set_algorithm(tss_search))
        self.algorithm_layout.addWidget(self.tss_button)

        self.similarity_group_box = QGroupBox("Similarity Metric")
        self.similarity_layout = QVBoxLayout()
        self.similarity_group_box.setLayout(self.similarity_layout)
        self.side_menu_layout.addWidget(self.similarity_group_box)

        self.mad_radio_button = QRadioButton("Mean Absolute Difference (MAD)")
        self.mad_radio_button.setChecked(True)
        self.mad_radio_button.toggled.connect(self.set_similarity_metric)
        self.similarity_layout.addWidget(self.mad_radio_button)

        self.ssd_radio_button = QRadioButton("Sum of Squared Differences (SSD)")
        self.ssd_radio_button.toggled.connect(self.set_similarity_metric)
        self.similarity_layout.addWidget(self.ssd_radio_button)

        self.block_size_input = QLineEdit()
        self.search_radius_input = QLineEdit()
        self.block_size_input.setPlaceholderText("Default: 16")
        self.search_radius_input.setPlaceholderText("Default: 8")

        form_layout = QFormLayout()
        form_layout.addRow("Block Size:", self.block_size_input)
        form_layout.addRow("Search Radius:", self.search_radius_input)
        self.side_menu_layout.addLayout(form_layout)

        self.video_exit_label = QLabel("Press ESC to close the application")
        self.video_layout.addWidget(self.video_exit_label)

        self.tracking_tab = QWidget()
        self.tracking_layout = QVBoxLayout(self.tracking_tab)
        self.tabs.addTab(self.tracking_tab, "Tracking")

        self.tracking_video_label = QLabel()
        self.tracking_video_label.setScaledContents(True)
        self.tracking_scroll_area = QScrollArea()
        self.tracking_scroll_area.setWidgetResizable(True)
        self.tracking_scroll_area.setWidget(self.tracking_video_label)
        self.tracking_layout.addWidget(self.tracking_scroll_area)

        self.load_tracking_button = QPushButton("Load Video to Track")
        self.load_tracking_button.clicked.connect(self.load_tracking_video)
        self.tracking_layout.addWidget(self.load_tracking_button)

        self.stop_tracking_button = QPushButton("Stop Tracking")
        self.stop_tracking_button.clicked.connect(self.stop_tracking_video)
        self.tracking_layout.addWidget(self.stop_tracking_button)

        self.resume_tracking_button = QPushButton("Start / Resume Tracking")
        self.resume_tracking_button.clicked.connect(self.resume_tracking_video)
        self.tracking_layout.addWidget(self.resume_tracking_button)

        self.tracking_progress_bar = QProgressBar()
        self.tracking_layout.addWidget(self.tracking_progress_bar)

        self.tracking_exit_label = QLabel("Press ESC to close the application")
        self.tracking_layout.addWidget(self.tracking_exit_label)

        self.tracking_video_label.mousePressEvent = self.mouse_press_event_tracking
        self.tracking_video_label.mouseMoveEvent = self.mouse_move_event_tracking
        self.tracking_video_label.mouseReleaseEvent = self.mouse_release_event_tracking

    def set_similarity_metric(self):
        if self.mad_radio_button.isChecked():
            self.similarity_metric = "MAD"
        elif self.ssd_radio_button.isChecked():
            self.similarity_metric = "SSD"
        if self.video_processor:
            self.video_processor.similarity_metric = self.similarity_metric

    def load_tracking_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Tracking Video File", "",
                                                   "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)
        if file_path:
            self.video_path = file_path
            video_title = file_path.split("/")[-1]
            self.statusBar().showMessage(f"{video_title} was successfully loaded for tracking!")
            self.tracking_progress_bar.setValue(0)

            # Display the first frame to draw bounding box
            self.cap = cv2.VideoCapture(file_path)
            frame_read, frame = self.cap.read()

            if frame_read:
                height, width, _ = frame.shape
                self.resize(width + (int(width * 0.05)), height + (
                    int(height * 0.25)))  # resize by video's width and height, but also account for external UI elements
                self.tracking_video_label.setFixedSize(width, height)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_tracking_frame(frame_rgb)

    def start_tracking(self):
        if self.bounding_box is not None:
            if self.tracking_processor:
                self.tracking_processor.stop()
            self.tracking_processor = TrackingProcessor(self.video_path)
            self.tracking_processor.set_bounding_box(self.bounding_box)
            self.tracking_processor.frame_ready.connect(self.update_tracking_frame)
            self.tracking_processor.tracking_progress_updated.connect(self.update_tracking_progress)
            self.tracking_processor.start()
            self.tracking_started = True

    def update_tracking_frame(self, frame):
        try:
            self.current_frame = frame  # Save current frame for redrawing
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qImg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.tracking_video_label.setPixmap(QPixmap.fromImage(qImg))
        except Exception as e:
            print(f"Error updating tracking frame: {e}")

    def update_tracking_progress(self, current_frame, total_frames):
        self.tracking_progress_bar.setMaximum(total_frames)
        self.tracking_progress_bar.setValue(current_frame)
        self.tracking_progress_bar.setFormat(
            f"    Frames Processed: {current_frame} out of {total_frames} Total Frames")

    def stop_tracking_video(self):
        if self.tracking_processor:
            self.tracking_processor.stop()
            self.statusBar().showMessage("Tracking video stopped.")
            # Todo: Reset video's current frame to be the first frame in the video

    def set_algorithm(self, algorithm):
        if not self.video_path:
            QMessageBox.warning(self, "No Video Loaded", "Please load a video before selecting an algorithm.")
            return
        self.algorithm = algorithm
        try:
            block_size = int(self.block_size_input.text())
            search_radius = int(self.search_radius_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Values are automatically being set to default")
            block_size = 16
            search_radius = 8

        if self.video_processor:
            self.video_processor.stop()
        self.video_processor = VideoProcessor(self.video_path, self.algorithm, block_size, search_radius, self.similarity_metric)
        self.video_processor.frame_ready.connect(self.update_frame)  # Connect to the frame_ready signal
        self.video_processor.progress_updated.connect(self.update_progress)  # Connect to the progress_updated signal
        self.video_processor.start()

    def load_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
                                                   "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)
        if file_path:
            self.video_path = file_path
            video_title = file_path.split("/")[-1]
            self.statusBar().showMessage(f"{video_title} was successfully loaded!")
            self.progress_bar.setValue(0)

            cap = cv2.VideoCapture(file_path)
            frame_read, frame = cap.read()

            if frame_read:
                height, width, _ = frame.shape
                self.resize(width + (int(width * 0.33)), height + (
                    int(height * 0.25)))  # resize by video's width and height, but also account for external UI elements
                self.video_label.setFixedSize(width, height)

            if self.algorithm:
                self.set_algorithm(self.algorithm)

    def stop_video(self):
        if self.video_processor:
            self.video_processor.stop()
            self.statusBar().showMessage("Motion Estimation stopped.")

    def resume_video(self):
        if self.video_processor:
            self.video_processor.resume()
            self.statusBar().showMessage("Motion Estimation resumed.")

    def update_frame(self, frame):
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qImg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qImg))
        except Exception as e:
            print(f"Error updating frame: {e}")

    def mouse_press_event_tracking(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()  # Ensure end_point is initialized
            self.drawing = True

    def mouse_move_event_tracking(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.redraw_tracking_frame()

    def mouse_release_event_tracking(self, event):
        if event.button() == Qt.LeftButton:
            self.end_point = event.pos()
            self.drawing = False
            self.bounding_box = self.get_bounding_box()
            if self.tracking_processor and not self.tracking_processor.is_running:
                self.tracking_processor.draw_bounding_box(self.bounding_box)
            self.redraw_tracking_frame()

    def redraw_tracking_frame(self):
        if self.current_frame is not None and self.start_point and self.end_point:
            frame_copy = self.current_frame.copy()
            cv2.rectangle(frame_copy, (self.start_point.x(), self.start_point.y()),
                          (self.end_point.x(), self.end_point.y()), (255, 0, 0), 2)
            height, width, channel = frame_copy.shape
            bytes_per_line = 3 * width
            qImg = QImage(frame_copy.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.tracking_video_label.setPixmap(QPixmap.fromImage(qImg))

    def get_bounding_box(self):
        if self.start_point and self.end_point:
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        return None

    def resume_tracking_video(self):
        if not self.tracking_started:
            self.start_tracking()
            self.statusBar().showMessage("Tracking started.")
        elif self.tracking_processor:
            self.tracking_processor.resume()
        self.statusBar().showMessage("Tracking resumed.")

    def update_progress(self, current_frame, total_frames):
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(current_frame)
        self.progress_bar.setFormat(f"    Frames Processed: {current_frame} out of {total_frames} Total Frames")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close_application()

    def close_application(self):
        if self.video_processor:
            self.video_processor.stop()
        if self.tracking_processor:
            self.tracking_processor.stop()
        self.close()

    def closeEvent(self, event):
        self.close_application()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    visualizer = MotionVectorVisualizer()
    visualizer.setMinimumSize(480, 320)
    visualizer.show()
    sys.exit(app.exec_())
