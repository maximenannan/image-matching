from typing import Tuple

import numpy as np


def pick_triangles(points: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a numpy ndarray of points of shape (N,2) this function extracts t triangles from that
    containing the frist points, t triangles containing the second one and so forth so on.
    :param points: numpy ndarray of points (N,2)
    :param t: number of triangles to extract by point
    :return: numpy ndarray of shape (N,t,3,2)
    -> first dimension == points
    -> second dimension == triangles for one  point
    --> thirs dimension == points of the triangles
    --> 4th dimension == coordinates of the point
    and a  numpy ndarray of shape (N,t,3,2) containing the indexes of the selected triangles of
    shape (N,t,3) --> third dimension is the indexes of the triangle
    """
    selected_triangles_index = []
    N = points.shape[0]
    for i in range(N):
        indexes = np.array(
            [np.random.choice(np.arange(1, N), size=2, replace=False) for _ in range(t)])
        indexes = (indexes + i) % N
        indexes = np.concatenate([i * np.ones((t, 1), dtype='int'), indexes], axis=1)
        selected_triangles_index.append(indexes)
    selected_triangles_index = np.vstack(selected_triangles_index)
    selected_triangles = points[selected_triangles_index].reshape((-1, t, 3, 2))
    selected_triangles_index = selected_triangles_index.reshape((-1, t, 3))
    return selected_triangles, selected_triangles_index


def extract_all_triangles(points: np.ndarray) -> np.ndarray:
    """
    Given a numpy array of points returns all the triangles that can be built with those points
    :param points: numpy array of points (shape = (N,2))
    :return: numpy array of triangles of shape (N, N, N, 3, 2)
    """
    N = points.shape[0]
    all_indexes = np.array([(i, j, k) for i in range(N) for j in range(N) for k in range(N)])
    return points[all_indexes].reshape((N, N, N, 3, 2))

