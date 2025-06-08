import os

def process_line(line):
    parts = line.strip().split()
    if not parts:
        return line  # skip empty lines

    try:
        identifier = int(parts[0]) - 1
        if identifier < 0:
            identifier = 12
        elif identifier <= 5:
            identifier += 6
        elif identifier >= 6:
            identifier -= 6
        parts[0] = str(identifier)
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
    folder_path = r'D:\Projects\Uni\Chessy3D\data\roboflow-dataset'  # Change this to the path of your folder

    for relative in ['train/labels', 'test/labels', 'valid/labels']:
        process_files_in_folder(f'{folder_path}/{relative}')
