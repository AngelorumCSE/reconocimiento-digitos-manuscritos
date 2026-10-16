"""El SSIM vectorizado da lo mismo que el de scikit-image."""

import numpy as np
from skimage.metrics import structural_similarity

from src.ssim import ssim_por_pares


def test_coincide_con_skimage():
    """El SSIM de cada pareja coincide con el de scikit-image y vale 1 en la pareja idéntica."""
    rng = np.random.default_rng(0)
    A = rng.random((4, 28, 28)).astype(np.float32)
    B = rng.random((5, 28, 28)).astype(np.float32)
    B[0] = A[0]  # una pareja idéntica
    S = ssim_por_pares(A, B)
    referencia = np.array([[structural_similarity(a, b, data_range=1.0) for b in B] for a in A])
    assert S.shape == (4, 5)
    assert np.abs(S - referencia).max() < 1e-5
    assert abs(S[0, 0] - 1.0) < 1e-6


def test_bloques_no_cambian_el_resultado():
    """Calcular por bloques más pequeños da el mismo resultado."""
    rng = np.random.default_rng(1)
    A = rng.random((7, 28, 28)).astype(np.float32)
    B = rng.random((9, 28, 28)).astype(np.float32)
    assert np.allclose(ssim_por_pares(A, B, bloque_a=2, bloque_b=4), ssim_por_pares(A, B), atol=1e-6)
