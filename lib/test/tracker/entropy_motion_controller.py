"""Small, testable primitives for causal entropy-conditioned box mixing."""

import math

import numpy as np
import torch


def normalized_probability_entropy(probabilities):
    """Mean normalized entropy of already-normalized coordinate probabilities.

    The returned scalar stays on the probabilities device.  Callers must defer host
    materialization until the released tracker already materializes its box.
    """
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy.mean() / math.log(float(probabilities.shape[-1]))


def constant_velocity_prior(last_box, previous_box):
    """One-step causal xywh extrapolation from two accepted tracker states."""
    last = np.asarray(last_box, dtype=np.float64)
    previous = np.asarray(previous_box, dtype=np.float64)
    if last.shape != (4,) or previous.shape != (4,):
        raise ValueError("last_box and previous_box must each be xywh length four")
    return last + (last - previous)


def mix_xywh(model_box, motion_box, alpha):
    """Convexly mix a model xywh state and a causal motion xywh state."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    model = np.asarray(model_box, dtype=np.float64)
    motion = np.asarray(motion_box, dtype=np.float64)
    return (1.0 - alpha) * model + alpha * motion
