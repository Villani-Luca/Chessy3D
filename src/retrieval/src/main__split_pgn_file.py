import os
import pathlib

ROOT = pathlib.Path.cwd().parent.parent.parent
PGN_FOLDER = (ROOT / "data/retrieval/lumbrasgigabase").resolve()
OUTPUT_FOLDER = (ROOT / 'data/retrieval/lumbrasgigabase/splitted').resolve()

SPLIT_SIZE = 5000
ONLY_ONE = False
TEMP_FILE_NAME = 'temp_lumbras'

def split_large_file(input_path, output_path, output_prefix='split_part', starting_number = 0, starting_file_number = 1):
    file_counter = starting_file_number
    filename = input_path.name.split('.')[0]
    temp_filename = f"{output_path}/{TEMP_FILE_NAME}"

    output_file = open(temp_filename, "w", encoding="utf-8")
    total_count = starting_number

    count = 0

    with input_path.open('r', encoding='utf-8') as f:
        prev_line_empty = False

        for line in f:
            # If previous line was empty and this line starts with a word
            if prev_line_empty and line.startswith('['):
                # Start a new output file
                count += 1
                total_count += 1

                if count % SPLIT_SIZE == 0:
                    output_file.close()
                    os.rename(temp_filename, f"{output_path}/{file_counter}_{count}_{total_count}_lumbras.pgn")
                    if ONLY_ONE:
                        break

                    file_counter += 1
                    count = 0
                    output_file = open(temp_filename, "w", encoding="utf-8")

            output_file.write(line)
            prev_line_empty = (line.strip() == '')

    output_file.close()
    os.rename(temp_filename, f"{output_path}/{file_counter}_{count}_{total_count}_lumbras.pgn")

    return (total_count, file_counter)


# Example usage

if __name__ == '__main__':
    file_counter = 1
    record_count = 0

    for pgnfile in PGN_FOLDER.glob('*.pgn'):
        print(f'Processing {pgnfile.name}')
        result = split_large_file(pgnfile, OUTPUT_FOLDER.as_posix(), starting_number=record_count, starting_file_number=file_counter)
        record_count += result[0]
        file_counter += result[1]
