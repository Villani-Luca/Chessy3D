from enum import StrEnum
from typing import Callable
import cv2
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PySide6.QtCore import Qt

class FileUploader(QWidget):
    class Tabs(StrEnum):
        ORIGINAL = "Original"
        CANNY = "Canny"
        HOUGH = "Hough"
        SQUARES = "Squares"
        FINAL = "Final"

    def __init__(self, callback: Callable[['FileUploader', str], None] | None = None):
        super().__init__()
        self.file_upload_callback = callback

        # Layout to hold the tab widget
        layout = QVBoxLayout(self)

        # Tab widget with four tabs
        self.tabs = QTabWidget(self)
        self.tab_names = [e.value for e in FileUploader.Tabs]
        self.image_labels = {}

        for name in self.tab_names:
            label = QLabel("Drag and Drop a File Here")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 2px dashed #aaa;")
            label.setAcceptDrops(False)  # Disable drop for individual labels
            self.tabs.addTab(label, name.capitalize())
            self.image_labels[name] = label

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.setAcceptDrops(True)  # Enable drop for the whole widget

    def set_opencv_image(self, img: cv2.typing.MatLike, tab: Tabs):
        tab_name = tab.value
        if tab_name not in self.tab_names:
            raise ValueError(f"Invalid tab name: {tab_name}. Must be one of: {self.tab_names}")

        # Convert the OpenCV image to QImage and display in the specified tab
        color = not (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1))
        height, width = img.shape[:2]
        q_image = QImage(
            img.data,
            width,
            height,
            img.strides[0],
            QImage.Format.Format_BGR888 if color else QImage.Format.Format_Grayscale8
        )

        pixmap = QPixmap.fromImage(q_image).scaled(self.size())
        self.image_labels[tab_name].setPixmap(pixmap)
        self.image_labels[tab_name].setText("")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                pixmap = QPixmap(file_path).scaled(self.size())

                self.image_labels[FileUploader.Tabs.ORIGINAL.value].setPixmap(pixmap)
                self.image_labels[FileUploader.Tabs.ORIGINAL.value].setText("")  # Remove placeholder text

                if self.file_upload_callback is not None:
                    self.file_upload_callback(self, file_path)

            elif file_path.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
                for name in self.tab_names:
                    self.image_labels[name].setText("Video support is pending!")
