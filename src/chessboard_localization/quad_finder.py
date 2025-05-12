import numpy as np
import cv2
import drawing as dw


# original version to be optimized
def find_quads(intersections):
    """Finds quads from segments intersections.
    This must be used for images, so the reference system is the top left corner and negative coordinates are now allowed.

    The input must be a numpy array of shape (N, M, 2) of dtype np.int32:
    - N represents the number of vertical-ish segments, M represents the number of horizontal-ish segments.
    - The third dimension represents the intersection point.

    If two segments are not intersecting, the point will contain [-1, -1] (sentinel).
    This is ok because negative coordinates are not allowed in this context.

    The output is a numpy array of shape (K, 4, 2) of dtype np.int32:
    - The first dimension is the number o quads.
    - The second dimension is the number of points
    - The third dimension is the point coordinate.
    """

    # all elements where the last dimension (third) doens't have all values = -1
    # valid_intersections_mask = np.all(intersections != -1, axis = -1)
    # valid_intersections = intersections[valid_intersections_mask]

    N = intersections.shape[0]
    M = intersections.shape[1]

    quads_list = []

    # the last row and the last column are not considered because they are handled from previous cases
    for i in range(N - 1):
        for j in range(M - 1):
            # fixed_point = intersections[i, j]
            fixed_point_index = np.array((i, j))

            # removing i here is an optimization to avoid considering already seen cases
            columns_indexes = np.zeros((N - i - 1, 2), dtype=np.int32)
            columns_row_selector = i + 1
            columns_column_selector = j
            columns_indexes[:, 0] = np.arange(columns_row_selector, N)
            columns_indexes[:, 1] = columns_column_selector
            # columns_points = intersections[columns_row_selector :, columns_column_selector]
            number_of_columns = len(columns_indexes)

            # removing j here is an optimization to avoid considering already seen cases
            row_indexes = np.zeros((M - j - 1, 2), dtype=np.int32)
            rows_row_selector = i
            rows_column_selector = j + 1
            row_indexes[:, 0] = rows_row_selector
            row_indexes[:, 1] = np.arange(rows_column_selector, N)
            # row_points = intersections[rows_row_selector, rows_column_selector :]
            number_of_rows = len(row_indexes)

            # fixed point repeated  number_column_points * number_row_points times
            first_column = np.repeat(
                [fixed_point_index], number_of_columns * number_of_rows, axis=0
            )

            # number of columns repeated number_row_points times
            second_column = np.repeat(columns_indexes, number_of_rows, axis=0)

            # number of rows tiled (alternated) number_of_columns times
            third_column = np.tile(row_indexes, (number_of_columns, 1))

            # combination of the coordinates of the second and third column to find a point in common
            forth_column = np.hstack((second_column[:, 0:1], third_column[:, 1:2]))

            # stack the columns into a combined array
            combinations = np.stack(
                [first_column, second_column, third_column, forth_column], axis=1
            )

            # add quad to the array
            quads_list.append(combinations)

    # Shape (K, 4, 2)
    quads_indexes = np.concatenate(quads_list, axis=0)
    quadr_coordinates = intersections[quads_indexes[:, :, 0], quads_indexes[:, :, 1], :]

    # discard quads with at least a [-1, -1]
    negative_mask = ~np.all(quadr_coordinates == -1, axis=-1).any(axis=-1)
    filtered_quads_coordinates = quadr_coordinates[negative_mask]

    # discard duplicates (maybe not needed)
    unique_quads = np.unique(filtered_quads_coordinates, axis=0)

    # ordered quad vertices
    ordered_quads = np.array(
        [cv2.convexHull(quad).reshape(4, 2) for quad in unique_quads],
        dtype=np.int32,
    )

    return ordered_quads


# original version
def filter_squares(quads, rtol=0.20, atol=20.0):

    squares = []

    for quad in quads:
        p1, p2, p3, p4 = quad

        d12 = np.linalg.norm(p1 - p2)
        d23 = np.linalg.norm(p2 - p3)
        d34 = np.linalg.norm(p3 - p4)
        d31 = np.linalg.norm(p4 - p1)

        if not np.isclose(d12, d34, rtol, atol):
            continue

        if not np.isclose(d23, d31, rtol, atol):
            continue

        if not np.isclose(d12, d23, rtol, atol):
            continue

        squares.append(quad)

    return np.array(squares)


# optimized version
def filter_squares_fast(quads, rtol=0.20, atol=20.0):
    edges = np.roll(quads, shift=-1, axis=1) - quads
    edge_lengths = np.linalg.norm(edges, axis=2)

    opposite_eq = np.isclose(
        edge_lengths[:, 0], edge_lengths[:, 2], rtol, atol
    ) & np.isclose(edge_lengths[:, 1], edge_lengths[:, 3], rtol, atol)
    adjacent_eq = np.isclose(edge_lengths[:, 0], edge_lengths[:, 1], rtol, atol)
    is_square = opposite_eq & adjacent_eq

    return quads[is_square]


def __draw_quads(intersections, quads, height=512, width=512):

    points = intersections.reshape(-1, 2)

    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    for x, y in points:
        cv2.circle(black_image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    for quad in quads:
        quad_pts = quad.reshape((-1, 1, 2))
        cv2.polylines(
            black_image, [quad_pts], isClosed=True, color=(0, 255, 0), thickness=1
        )

    dw.display_image_cv2(black_image)


def ___simple_case():

    # 3 vertical segments, 3 horizontal segments, 9 intersection points (3, 3, 2)
    # [-1, -1] is a missing intersection

    intersections = np.array(
        [
            [[100, 100], [100, 200], [100, 300]],
            [[200, 100], [200, 200], [200, 300]],
            [[300, 100], [-1, -1], [300, 300]],
        ],
        dtype=np.int32,
    )

    quads = find_quads(intersections)

    squares = filter_squares_fast(quads)

    __draw_quads(intersections, squares)


def ___perspective_distorsion_case1():

    intersections = np.array(
        [
            [[100, 100], [110, 200], [120, 300]],
            [[200, 100], [200, 200], [210, 300]],
            [[300, 100], [-1, -1], [280, 300]],
        ],
        dtype=np.int32,
    )

    quads = find_quads(intersections)
    squares = filter_squares_fast(quads)
    # squares = [squares[0]]

    __draw_quads(intersections, squares)


def ___perspective_distorsion_case2():

    intersections = np.array(
        [
            [[80, 90], [85, 200], [90, 310], [95, 420], [100, 530]],
            [[180, 80], [185, 190], [190, 300], [195, 410], [200, 520]],
            [[280, 70], [290, 180], [-1, -1], [300, 400], [310, 510]],
            [[380, 90], [370, 200], [360, 310], [350, 420], [340, 530]],
            [[480, 100], [470, 210], [460, 320], [-1, -1], [440, 540]],
        ],
        dtype=np.int32,
    )

    quads = find_quads(intersections)
    squares = filter_squares_fast(quads, 0.05, 5)

    __draw_quads(intersections, squares, 1024, 1024)


if __name__ == "__main__":
    ___simple_case()
    ___perspective_distorsion_case1()
    ___perspective_distorsion_case2()
