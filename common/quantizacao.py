import numpy as np
from typing import List, Tuple

def quantize32to16(
    weights: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    q_weights   = []
    scales      = []
    zero_points = []

    q_min, q_max = -32768, 32767

    for w in weights:
        min_val = float(np.min(w))
        max_val = float(np.max(w))

        if max_val == min_val:
            scale      = 1.0
            zero_point = q_min
        else:
            scale      = (max_val - min_val) / (q_max - q_min)
            zero_point = int(np.round(q_min - min_val / scale))

        zero_point = int(np.clip(zero_point, q_min, q_max))

        q = np.round(w / scale + zero_point).astype(np.int16)

        q_weights.append(q)
        scales.append(scale)
        zero_points.append(zero_point)

    return q_weights, scales, zero_points


def dequantize16to32(
    qweights: List[np.ndarray],
    scales:   List[float],
    zero_points: List[int]
) -> List[np.ndarray]:
    deq = []
    for q, scale, zp in zip(qweights, scales, zero_points):
        arr = (q.astype(np.float32) - zp) * scale
        deq.append(arr)
    return deq

def quantize32to8(
    weights: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """
    Quantização linear de float32 para int8.
    Mapeia cada tensor [min_val, max_val] → [−128, 127].
    Retorna:
      - lista de arrays quantizados (np.int8)
      - lista de scales (float)
      - lista de zero_points (int)
    """
    q_weights   = []
    scales      = []
    zero_points = []

    q_min, q_max = -128, 127

    for w in weights:
        min_val = float(np.min(w))
        max_val = float(np.max(w))

        if max_val == min_val:
            scale      = 1.0
            zero_point = q_min
        else:
            scale      = (max_val - min_val) / (q_max - q_min)
            zero_point = int(np.round(q_min - min_val / scale))

        zero_point = int(np.clip(zero_point, q_min, q_max))
        q = np.round(w / scale + zero_point).astype(np.int8)

        q_weights.append(q)
        scales.append(scale)
        zero_points.append(zero_point)

    return q_weights, scales, zero_points


def dequantize8to32(
    qweights: List[np.ndarray],
    scales:   List[float],
    zero_points: List[int]
) -> List[np.ndarray]:
    deq = []
    for q, scale, zp in zip(qweights, scales, zero_points):
        arr = (q.astype(np.float32) - zp) * scale
        deq.append(arr)
    return deq