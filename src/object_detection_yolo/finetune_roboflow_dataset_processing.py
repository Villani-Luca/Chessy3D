import os

convert_to_rf = {
    '0': '6',    # black-bishop -> black-bishop
    '1': '7',    # black-king -> black-king
    '2': '8',    # black-knight -> black-knight
    '3': '9',    # black-pawn -> black-pawn
    '4': '10',   # black-queen -> black-queen
    '5': '11',   # black-rook -> black-rook
    '6': '0',    # white-bishop -> white-bishop
    '7': '1',    # white-king -> white-king
    '8': '2',    # white-knight -> white-knight
    '9': '3',    # white-pawn -> white-pawn
    '10': '4',   # white-queen -> white-queen
    '11': '5',   # white-rook -> white-rook
    '12': '12'   # empty -> empty
}

def process_line(line):
    parts = line.strip().split()
    if not parts:
        return line  # skip empty lines

    try:
        parts[0] = convert_to_rf[parts[0]]
        return ' '.join(parts) + '\n'
    except ValueError:
        # In case the line doesn't start with an integer, leave it unchanged
        return line

def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            with open(file_path, 'w') as f:
                for line in lines:
                    f.write(process_line(line))

if __name__ == '__main__':
    folder_path = r'D:\CodeProjects\University\Chessy3d\Chessy3D\data\chess_detection_dataset_270625'  # Change this to the path of your folder

    for relative in ['train/labels', 'test/labels', 'valid/labels']:
        process_files_in_folder(f'{folder_path}/{relative}')
