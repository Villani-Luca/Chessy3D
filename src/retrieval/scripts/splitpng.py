import os

INPUT_PATH = "D:/uni/Chessy3D_Lumbras/LumbrasGigaBase 2020.pgn"
OUTPUT_FOLDER = "D:/uni/Chessy3D_Lumbras/splitted"
SPLIT_SIZE = 100
ONLY_ONE = True

def split_large_file(input_path, output_path, output_prefix='split_part'):
    file_counter = 1
    filename = os.path.basename(input_path)
    output_file = open(f"{output_path}/{filename}_{file_counter}.pgn", "w", encoding="utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        count = 0
        prev_line_empty = False

        for line in f:
            # If previous line was empty and this line starts with a word
            if prev_line_empty and line.startswith('['):
                # Start a new output file
                count += 1

                if count % SPLIT_SIZE == 0:
                    output_file.close()
                    if ONLY_ONE:
                        break

                    file_counter += 1
                    output_file = open(f"{output_path}/{filename}_{file_counter}.pgn", "w", encoding="utf-8")

            output_file.write(line)
            prev_line_empty = (line.strip() == '')

    output_file.close()

# Example usage

split_large_file(INPUT_PATH, OUTPUT_FOLDER)
