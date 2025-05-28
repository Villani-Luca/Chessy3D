from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QLabel, QVBoxLayout, QTableWidgetItem


class DataGrid(QTableWidget):
    def __init__(self):
        super().__init__()  # Example: 5 rows, 3 columns

        self.setHorizontalHeaderLabels(["Column 1", "Column 2", "Column 3"])

        # Set up the label to display when the table is empty
        self.empty_label = QLabel("Waiting for image upload")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("font-size: 20px; font-weight: bold; color: gray;")
        self.empty_label.setVisible(True)  # Initially visible

        # Set up a layout to hold the table and the label
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.empty_label)
        self.setLayout(self.layout)

        # Fill data
        self.update_empty_label()

    def update_empty_label(self):
        # If the table has data, hide the "waiting" label
        if self.rowCount() > 0 and self.columnCount() > 0:
            self.empty_label.setVisible(False)
        else:
            self.empty_label.setVisible(True)

    def set_data(self, data):
        # Populate the table with data, you can modify this method based on how you want to add data
        for row in range(len(data)):
            for col in range(len(data[row])):
                self.setItem(row, col, QTableWidgetItem(data[row][col]))

        self.update_empty_label()
