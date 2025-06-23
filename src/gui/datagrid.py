from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QLabel, QVBoxLayout, QTableWidgetItem, QHBoxLayout, QPushButton, QWidget


class DataGrid(QTableWidget):
    def __init__(self, on_refresh_button, on_row_double_click, headers=None):
        if headers is None:
            headers = ["Id", "Event", "Date", "White", "White title", "Black", "Black title", "Distance"]

        super().__init__()  # Example: 5 rows, 3 columns

        # self.setColumnCount(3)
        self.on_row_double_click = on_row_double_click
        self.headers = headers
        self.setHorizontalHeaderLabels(headers)

        # Set up the label to display when the table is empty
        self.empty_label = QLabel("Waiting for image upload")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("font-size: 20px; font-weight: bold; color: gray;")
        self.empty_label.setVisible(True)  # Initially visible

        # Create the button and connect it to a function
        self.top_right_button = QPushButton("Refresh", parent=self)
        self.top_right_button.move(self.width() - 110, 10)
        self.top_right_button.clicked.connect(on_refresh_button)

        # Layout for the button aligned to the right
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        button_layout.addWidget(self.top_right_button)

        # Set up a layout to hold the table and the label
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.empty_label)
        self.setLayout(self.layout)

        # Connect the double-click signal to callback
        self.cellDoubleClicked.connect(self.handle_row_double_click)

        # Fill data
        self.update_empty_label()

    def update_empty_label(self):
        # If the table has data, hide the "waiting" label
        if self.rowCount() > 0 and self.columnCount() > 0:
            self.empty_label.setVisible(False)
        else:
            self.empty_label.setVisible(True)

    def set_enable_refreshbutton(self, state):
        self.top_right_button.setEnabled(state)

    def set_data(self, data):
        # Populate the table with data, you can modify this method based on how you want to add data
        # print(data)

        self.reset()
        self.clearContents()
        self.setColumnCount(len(self.headers))
        self.setHorizontalHeaderLabels(self.headers)
        self.setRowCount(len(data))

        for row in  range(len(data)):
            for col in range(len(data[row])):
                self.setItem(row, col, QTableWidgetItem(str(data[row][col])))

        self.update_empty_label()

    def handle_row_double_click(self, row, column):
        # Get all data from the clicked row and pass it to the callback
        row_data = [self.item(row, col).text() if self.item(row, col) else "" for col in range(self.columnCount())]
        self.on_row_double_click(row_data)