class ChessPiece:
    def __init__(self, x_min, y_min, x_max, y_max, xn_min, yn_min, xn_max, yn_max, class_number, class_name, class_confidence, class_color, xyxy_box_position, xyxy_n_box_position):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.xn_min = xn_min
        self.yn_min = yn_min
        self.xn_max = xn_max
        self.yn_max = yn_max
        self.class_number = class_number
        self.class_name = class_name
        self.class_confidence = class_confidence
        self.class_color = class_color
        self.xyxy_box_position = xyxy_box_position
        self.xyxy_n_box_position = xyxy_n_box_position
        print(">> ChessPiece constructor called")