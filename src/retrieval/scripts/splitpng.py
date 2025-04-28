import os
import pathlib

ROOT = pathlib.Path.cwd().parent.parent.parent
INPUT_PATH =  (ROOT / 'data/retrieval/lumbrasgigabase/lumbrasgigabase_2020.pgn').resolve()
OUTPUT_FOLDER = (ROOT / 'data/retrieval/lumbrasgigabase/splitted').resolve()

SPLIT_SIZE = 1000
ONLY_ONE = True

def split_large_file(input_path, output_path, output_prefix='split_part'):
    file_counter = 1
    filename = input_path.name.split('.')[0]
    output_file = open(f"{output_path}/{filename}_{file_counter}.pgn", "w", encoding="utf-8")

    with input_path.open('r') as f:
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

split_large_file(INPUT_PATH, OUTPUT_FOLDER.as_posix())
