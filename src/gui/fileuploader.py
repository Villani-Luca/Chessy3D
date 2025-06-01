from typing import Callable

import cv2.typing
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt


class FileUploader(QLabel):
    def __init__(self, callback: Callable[['FileUploader', str], None] | None = None):
        super().__init__("Drag and Drop a File Here")

        self.file_upload_callback = callback

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 2px dashed #aaa;")
        self.setAcceptDrops(True)

    def set_opencv_image(self, img: cv2.typing.MatLike):
        height, width, channels = img.shape
        q_image = QImage(img.data, width, height, channels * width, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image).scaled(self.size())
        self.setPixmap(pixmap)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                pixmap = QPixmap(file_path)
                self.setPixmap(pixmap.scaled(self.size()))

                if self.file_upload_callback is not None:
                    self.file_upload_callback(self, file_path)

            elif file_path.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
                self.setText("Video support is pending!")