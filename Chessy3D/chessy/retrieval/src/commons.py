
def zobrist_to_s64(zobrist: int):
    return zobrist if zobrist <= 9223372036854775807 else zobrist - 18446744073709551616