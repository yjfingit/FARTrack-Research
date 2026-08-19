"""Host-side causal primitives for branch-disagreement state mixing."""

import numpy as np


def constant_velocity_prior(last_box, previous_box):
    """One-step xywh extrapolation using only previously accepted states."""
    last = np.asarray(last_box, dtype=np.float64)
    previous = np.asarray(previous_box, dtype=np.float64)
    if last.shape != (4,) or previous.shape != (4,):
        raise ValueError("last_box and previous_box must each be xywh length four")
    return last + (last - previous)


def branch_l1_disagreement(sequence_box, feature_box):
    """Mean absolute difference over the four released normalized coordinates."""
    sequence = np.asarray(sequence_box, dtype=np.float64)
    feature = np.asarray(feature_box, dtype=np.float64)
    if sequence.shape != (4,) or feature.shape != (4,):
        raise ValueError("branch boxes must each contain four normalized coordinates")
    return float(np.abs(sequence - feature).mean())


def should_mix(disagreement, threshold):
    """Strictly activate only above the frozen disagreement threshold."""
    return float(disagreement) > float(threshold)


def mix_xywh(model_box, motion_box, alpha):
    """Convex image-space xywh mix; alpha=0 exactly retains the model box."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    model = np.asarray(model_box, dtype=np.float64)
    motion = np.asarray(motion_box, dtype=np.float64)
    return (1.0 - alpha) * model + alpha * motion
