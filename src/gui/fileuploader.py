from enum import StrEnum
from typing import Callable
import cv2
from PyQt6.QtCore import QSize
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget, QFileDialog, QPushButton
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
        self.tab_names = list([e for e in FileUploader.Tabs])
        self.image_labels: dict[FileUploader.Tabs, QLabel] = {}
        self.download_buttons: dict[FileUploader.Tabs, QPushButton] = {}

        for name in self.tab_names:
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            label = QLabel("Drag and Drop a File Here")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 2px dashed #aaa;")
            label.setAcceptDrops(False)  # Disable drop for individual labels
            label.setFixedSize(550, 550)

            download_button = QPushButton("Download Image")
            download_button.clicked.connect(lambda _, n=name: self.download_image(n))
            download_button.setEnabled(False)

            tab_layout.addWidget(label)
            tab_layout.addWidget(download_button)

            self.tabs.addTab(tab_widget, name.capitalize())
            self.image_labels[name] = label
            self.download_buttons[name] = download_button

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
        self.download_buttons[tab_name].setEnabled(True)

    def download_image(self, tab_name: str):
        label = self.image_labels[tab_name]
        pixmap = label.pixmap()
        if pixmap:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"{tab_name}.png", "Images (*.png *.jpg *.bmp)")
            if file_path:
                pixmap.save(file_path)

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
                size = self.image_labels[FileUploader.Tabs.ORIGINAL].size()
                pixmap = QPixmap(file_path).scaled(size)

                self.image_labels[FileUploader.Tabs.ORIGINAL].setPixmap(pixmap)
                self.image_labels[FileUploader.Tabs.ORIGINAL].setText("")  # Remove placeholder text
                self.download_buttons[FileUploader.Tabs.ORIGINAL].setEnabled(True)

                if self.file_upload_callback is not None:
                    self.file_upload_callback(self, file_path)

            elif file_path.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
                for name in self.tab_names:
                    self.image_labels[name].setText("Video support is pending!")
