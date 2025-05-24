import numpy as np

def convert_segments_to_hesse_normal_form_lines_alt(segments):
    # segments must have shape (N, 4) -> x1, y1, x2, y2

    hesse_normal_lines = np.zeros((len(segments), 2))

    x1 = segments[:, 0]
    y1 = segments[:, 1]
    x2 = segments[:, 2]
    y2 = segments[:, 3]

    # Calculate line coefficients: Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # Calculate norm (length of normal vector)
    norm = np.sqrt(A**2 + B**2)

    # Handle vertical lines (where x1 ≈ x2)
    vertical_mask = np.isclose(x1, x2)
    non_vertical_mask = ~vertical_mask

    # For non-vertical lines
    if np.any(non_vertical_mask):
        # Hesse normal form: rho = -C/norm (with sign correction)
        rho = (-C[non_vertical_mask]) / norm[non_vertical_mask]
        theta = np.arctan2(B[non_vertical_mask], A[non_vertical_mask])

        # Ensure rho is positive by adjusting theta if needed
        neg_rho_mask = rho < 0
        rho[neg_rho_mask] = -rho[neg_rho_mask]
        theta[neg_rho_mask] += np.pi

        # Normalize theta to [0, π) range
        theta = np.mod(theta, np.pi)

        hesse_normal_lines[non_vertical_mask, 0] = rho
        hesse_normal_lines[non_vertical_mask, 1] = theta

    # For vertical lines
    if np.any(vertical_mask):
        rho = np.abs(x1[vertical_mask])  # Distance from y-axis
        theta = np.pi / 2  # 90 degrees (normal vector points horizontally)

        # Adjust sign of rho based on which side of origin
        sign_correction = np.sign(x1[vertical_mask])
        rho *= sign_correction

        hesse_normal_lines[vertical_mask, 0] = rho
        hesse_normal_lines[vertical_mask, 1] = theta

    return hesse_normal_lines

def convert_segments_to_hesse_normal_form_lines(segments):
    # segments must have shape (N, 4) -> x1, y1, x2, y2

    hesse_normal_lines = np.zeros((len(segments), 2))
    
    x1, y1, x2, y2 = segments[:,0], segments[:,1], segments[:,2], segments[:,3]

    # Vectorized calculation for all segments
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2

    norm = np.sqrt(A**2 + B**2)
    epsilon = 1e-10  # Small value to avoid division by zero
    norm = np.where(norm < epsilon, epsilon, norm)  # Numerical stability

    # Default case (handles non-vertical lines)
    rho = -C / norm
    theta = np.arctan2(B, A)  # Preserves your angle convention

    # Special case for vertical lines (x1 ≈ x2)
    vertical_mask = np.abs(x2 - x1) < epsilon
    rho[vertical_mask] = x1[vertical_mask]  # Your original working version
    theta[vertical_mask] = np.pi/2  # 90° for vertical lines

    hesse_normal_lines[:,0] = rho
    hesse_normal_lines[:,1] = theta

    return hesse_normal_lines


def calculate_standard_line_coeffients_from_segments(segments):
    # segments must have shape (N, 4) -> x1, y1, x2, y2

    x1 = segments[:, 0]
    y1 = segments[:, 1]
    x2 = segments[:, 2]
    y2 = segments[:, 3]

    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1

    return np.stack((A, B, C), axis=1)


def calculate_line_slope_intercept_from_segments(segments):
    # segments must have shape (N, 4) -> x1, y1, x2, y2
    x1 = segments[:, 0]
    y1 = segments[:, 1]
    x2 = segments[:, 2]
    y2 = segments[:, 3]

    dx = x2 - x1
    dy = y2 - y1

    line_coefficients = np.zeros((len(segments), 2))

    vertical_mask = np.isclose(x1, x2)
    non_vertical_mask = ~vertical_mask

    if np.any(non_vertical_mask):
        m = dy[non_vertical_mask] / dx[non_vertical_mask]
        q = y1[non_vertical_mask] - m * x1[non_vertical_mask]
        line_coefficients[non_vertical_mask, 0] = m
        line_coefficients[non_vertical_mask, 1] = q

    if np.any(vertical_mask):
        line_coefficients[vertical_mask, 0] = np.inf
        line_coefficients[vertical_mask, 1] = x1[vertical_mask]

    return line_coefficients


def convert_hesse_normal_form_lines_to_segments(lines, segment_length=10000):
    # lines must have shape (N, 2) -> rho, theta

    segments = np.zeros((lines.shape[0], 4), dtype=np.int32)

    for index, line in enumerate(lines):
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + segment_length * (-b))
        y1 = int(y0 + segment_length * (a))
        x2 = int(x0 - segment_length * (-b))
        y2 = int(y0 - segment_length * (a))

        segments[index] = (x1, y1, x2, y2)

    return segments


# todo: modify to support multiple segments
def calculate_segments_intersection_point(segment1, segment2, epsilon=1e-10):
    # segments must have shape (4) -> x1, y1, x2, y2

    x1, y1, x2, y2 = segment1
    x3, y3, x4, y4 = segment2

    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    det = a1 * b2 - a2 * b1

    # Check if lines are parallel
    if abs(det) < epsilon:
        return None

    # Intersection point
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det

    x_in_seg1 = np.isclose(x, np.clip(x, min(x1, x2), max(x1, x2)), atol=epsilon)
    y_in_seg1 = np.isclose(y, np.clip(y, min(y1, y2), max(y1, y2)), atol=epsilon)
    x_in_seg2 = np.isclose(x, np.clip(x, min(x3, x4), max(x3, x4)), atol=epsilon)
    y_in_seg2 = np.isclose(y, np.clip(y, min(y3, y4), max(y3, y4)), atol=epsilon)

    if x_in_seg1 and y_in_seg1 and x_in_seg2 and y_in_seg2:
        return (x, y)

    return None


# todo: modify to support multiple lines
def calculate_hesse_normal_form_lines_intersection_point(
    line1, line2, max_height, max_width
):
    # lines must have shape (2) -> rho, theta

    rho1, theta1 = line1
    rho2, theta2 = line2

    cos1, sin1 = np.cos(theta1), np.sin(theta1)
    cos2, sin2 = np.cos(theta2), np.sin(theta2)

    # this is a cross product
    det = cos1 * sin2 - cos2 * sin1

    # Skip if there are no intersection (parallel lines)
    if np.isclose(det, 0, atol=1e-8):
        return None

    # Solving the linear system to find the intersection point
    A = np.array([[cos1, sin1], [cos2, sin2]])
    b = np.array([[rho1], [rho2]])

    try:
        x, y = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    x = int(np.round(x.item()))
    y = int(np.round(y.item()))

    # Drop points located outside the image
    if x < 0 or x >= max_width or y < 0 or y >= max_height:
        return None

    return (x, y)


def extend_segments_size(segments, image_height, image_width):

    extended_segments = np.zeros((segments.shape[0], 4), dtype=np.int32)

    for index, segment in enumerate(segments):
        x1, y1, x2, y2 = segment

        # Vertical line (m = 0)
        if x1 == x2:
            extended_segments[index] = (x1, 0, x1, image_height)
            continue

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Intersections with the left/right image sides
        x_left = 0
        x_right = image_width
        y_left = m * x_left + b
        y_right = m * x_right + b

        # Intersections with the top/bottom image sides
        y_top = 0
        y_bottom = image_height
        x_top = (y_top - b) / m
        x_bottom = (y_bottom - b) / m

        candidates = []

        # check if the left side intersection is in the image
        if y_left >= 0 and y_left <= image_height:
            candidates.append((x_left, y_left))

        # check if the right side intersection is in the image
        if y_right >= 0 and y_right <= image_height:
            candidates.append((x_right, y_right))

        # check if the top side intersection is in the image
        if x_top >= 0 and x_top <= image_width:
            candidates.append((x_top, y_top))

        # check if the bottom side intersection is in the image
        if x_bottom >= 0 and x_bottom <= image_width:
            candidates.append((x_bottom, y_bottom))

        # Normally there should be only 2 candidates (left/right or top/bottom)
        # but for perfectly diagonal line, the intersection points are the two corners of the image
        if len(candidates) >= 2:
            candidates = np.array(candidates)
            candidates = candidates[np.argsort(candidates[:, 0])]
            extended_segments[index] = (*candidates[0], *candidates[-1])
        else:
            # this should never happen
            extended_segments[index] = (x1, y1, x2, y2)

    return extended_segments


def expand_segments_relative(segments, expansion_amount, image_height, image_width):

    extended_segments = np.zeros((segments.shape[0], 4), dtype=np.int32)

    for index, segment in enumerate(segments):
        x1, y1, x2, y2 = segment

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1

        length = np.sqrt(dx**2 + dy**2)

        unit_dx = dx / length
        unit_dy = dy / length

        new_x1 = center_x - unit_dx * (length / 2 + expansion_amount)
        new_y1 = center_y - unit_dy * (length / 2 + expansion_amount)
        new_x2 = center_x + unit_dx * (length / 2 + expansion_amount)
        new_y2 = center_y + unit_dy * (length / 2 + expansion_amount)

        # clip segments to the image size
        clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clip_segment_to_image_size(
            new_x1, new_y1, new_x2, new_y2, image_height, image_width
        )

        extended_segments[index] = (
            int(np.round(clipped_x1)),
            int(np.round(clipped_y1)),
            int(np.round(clipped_x2)),
            int(np.round(clipped_y2)),
        )

    return extended_segments


def clip_segment_to_image_size(x1, y1, x2, y2, image_height, image_width):
    # Parametric line equations: x = x1 + t*(x2-x1), y = y1 + t*(y2-y1)
    t_values = [0.0, 1.0]  # Start with original segment

    # Calculate intersection with image boundaries
    for boundary in [0, image_width]:
        if (x2 - x1) != 0:  # Avoid division by zero
            t = (boundary - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
            if 0 <= y <= image_height:
                t_values.append(t)

    for boundary in [0, image_height]:
        if (y2 - y1) != 0:  # Avoid division by zero
            t = (boundary - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            if 0 <= x <= image_width:
                t_values.append(t)

    # Find the valid t-range that stays within image
    t_min = max(min(t_values), 0.0)
    t_max = min(max(t_values), 1.0)

    # Return clipped coordinates
    return (
        x1 + t_min * (x2 - x1),
        y1 + t_min * (y2 - y1),
        x1 + t_max * (x2 - x1),
        y1 + t_max * (y2 - y1),
    )
