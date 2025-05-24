import numpy as np
from sklearn.linear_model import RANSACRegressor
from collections import defaultdict

def perpendicular_distance(point, line_start, line_end):
    line = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    if np.linalg.norm(line) == 0:
        return np.linalg.norm(point_vec)
    proj = line * np.dot(point_vec, line) / np.dot(line, line)
    perp_vec = point_vec - proj
    return np.linalg.norm(perp_vec)

def compute_gamma(x1, x2, y1, y2):
    return 0.25 * (x1 + x2 + y1 + y2)

def compute_delta(a, b, t_delta):
    return (a + b) * t_delta

def length(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        pi, pj = self.find(i), self.find(j)
        if pi != pj:
            self.parent[pi] = pj

def fit_line(points):
    X = np.array(points)[:, 0].reshape(-1, 1)
    y = np.array(points)[:, 1]
    model = RANSACRegressor()
    model.fit(X, y)
    return model

# ========== Main Entry Function ==========

def link_and_merge_segments_array(segments_array, image_area, p=0.9):
    """
    segments_array: (N, 4) array where each row is [x1, y1, x2, y2]
    image_area: scalar (e.g., 250000 for a 500x500 image)
    p: threshold for similarity (e.g., 0.9)
    """
    n = segments_array.shape[0]
    uf = UnionFind(n)

    omega = (np.pi / 2) * (1 / (image_area ** 0.25))
    t_delta = p * omega

    for i in range(n):
        x1a, y1a, x2a, y2a = segments_array[i]
        A = (x1a, y1a)
        B = (x2a, y2a)
        a = length(A, B)

        for j in range(i + 1, n):
            x1b, y1b, x2b, y2b = segments_array[j]
            C = (x1b, y1b)
            D = (x2b, y2b)
            b = length(C, D)

            d1 = perpendicular_distance(A, C, D)
            d2 = perpendicular_distance(B, C, D)
            d3 = perpendicular_distance(C, A, B)
            d4 = perpendicular_distance(D, A, B)

            gamma = compute_gamma(d1, d2, d3, d4)
            delta = compute_delta(a, b, t_delta)

            if (a / gamma > delta) and (b / gamma > delta):
                uf.union(i, j)

    # Grouping
    groups = defaultdict(list)
    for idx in range(n):
        root = uf.find(idx)
        x1, y1, x2, y2 = segments_array[idx]
        groups[root].extend([(x1, y1), (x2, y2)])

    # Fit lines and return segments as [x1, y1, x2, y2]
    merged_segments = []
    for pts in groups.values():
        pts = np.array(pts)
        x_vals = pts[:, 0]
        y_vals = pts[:, 1]

        x_min, x_max = np.min(x_vals), np.max(x_vals)
        model = RANSACRegressor().fit(x_vals.reshape(-1, 1), y_vals)

        y_min = model.predict([[x_min]])[0]
        y_max = model.predict([[x_max]])[0]

        merged_segments.append([int(np.round(x_min)), int(np.round(y_min)), int(np.round(x_max)), int(np.round(y_max))])

    return merged_segments
