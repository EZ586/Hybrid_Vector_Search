# src/baselines/hybrid/early_stop.py
"""
Early-stop policies for hybrid IVF search.

This module is PURE logic: given the current search state, decide whether
we can stop. It does NOT talk to FAISS directly and does NOT read files.

We expose a common signature so the search loop can pick a policy by name.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SearchState = Dict[str, Any]
# expected keys (all optional except K and num_candidates):
# - "K": int
# - "num_candidates": int
# - "current_kth_score": Optional[float]
# - "probe_index": int            # 1-based or 0-based, but consistent
# - "global_bound": Optional[float]
# - "kth_history": List[float]    # prev kth scores, newest at end
# - "window": int                 # for stability-based stop
# - "epsilon": float              # tolerance for stability
# RM-style optional helpers (if provided by the search loop):
# - "neighbor_radius": Optional[float]      # E-th best score so far (E >= K)
# - "rm_window_median": Optional[float]     # median score of recent probes
# - "rm_window_size": int                   # window size used for rm_window_median


# ---------------------------------------------------------------------------
# Policy 1: stop when we have K candidates
# ---------------------------------------------------------------------------

def stop_when_k_reached(state: SearchState) -> Tuple[bool, Optional[str]]:
    """
    Simplest policy: stop as soon as we have at least K candidates.

    Args:
        state: dict-like search state, must contain:
            - "K": int
            - "num_candidates": int

    Returns:
        (should_stop, reason)
    """
    K = state.get("K")
    num_candidates = state.get("num_candidates", 0)
    if K is None:
        # malformed state, do not stop
        return False, None

    if num_candidates >= K:
        return True, "k_reached"
    return False, None


# ---------------------------------------------------------------------------
# Policy 2: stop when we have K AND kth score >= given global bound
# ---------------------------------------------------------------------------

def stop_when_k_and_bound(state: SearchState) -> Tuple[bool, Optional[str]]:
    """
    Stop only when:
      1) we have at least K candidates
      2) we know the current kth score
      3) we have a global bound
      4) kth_score >= global_bound

    If any needed piece is missing, do NOT stop.
    """
    K = state.get("K")
    num_candidates = state.get("num_candidates", 0)
    kth_score = state.get("current_kth_score", None)
    global_bound = state.get("global_bound", None)

    if K is None:
        return False, None

    if num_candidates < K:
        return False, None

    # if we don't have kth or don't have a bound, we cannot apply this policy
    if kth_score is None or global_bound is None:
        return False, None

    if kth_score >= global_bound:
        return True, "k_and_bound"
    return False, None


# ---------------------------------------------------------------------------
# Policy 3: stop when we have K AND kth score is stable (RM-lite)
# ---------------------------------------------------------------------------

def stop_when_k_and_stable(state: SearchState) -> Tuple[bool, Optional[str]]:
    """
    Stability-based policy: emulate "we are in phase 2".

    Stop when:
      - we have at least K candidates
      - we have run at least `min_probes` probes
      - we have a history of kth scores
      - over the last `window` probes, kth did not improve more than `epsilon`

    Expected state keys:
      - "K": int
      - "num_candidates": int
      - "kth_history": List[float]  (newest at end)
      - "probe_index": int          (1-based probe counter)
      - "window": int               (e.g. 3)
      - "epsilon": float            (e.g. 1e-3)
      - "min_probes": int           (e.g. 3)
    """
    K = state.get("K")
    num_candidates = state.get("num_candidates", 0)
    kth_history: List[float] = state.get("kth_history") or []
    probe_index: int = state.get("probe_index", 0)

    # Defaults: require at least 3 probes and use a 3-step stability window
    window: int = state.get("window", 3)
    epsilon: float = state.get("epsilon", 1e-3)
    min_probes: int = state.get("min_probes", 3)

    if K is None or num_candidates < K:
        return False, None

    # Do not allow early-stop before we've run enough probes
    if probe_index < min_probes:
        return False, None

    # need at least `window` points to judge stability
    if len(kth_history) < window:
        return False, None

    # compare oldest vs newest in the window
    recent = kth_history[-window:]
    oldest = recent[0]
    newest = recent[-1]

    # if newest is not much better than oldest, we say it's stable
    if (newest - oldest) <= epsilon:
        return True, "k_and_stable"

    return False, None

# ---------------------------------------------------------------------------
# Policy 4: stop when K AND RM-style "outside neighborhood" condition holds
# ---------------------------------------------------------------------------

def stop_when_rm_and_k(state: SearchState) -> Tuple[bool, Optional[str]]:
    """
    Relaxed-monotonicity-inspired policy.

    Intuition (for inner-product / similarity scores):
      - "neighbor_radius" is the E-th best score over all candidates so far
        (E >= K), so higher = better neighbor.
      - "rm_window_median" is the median score of recently returned vectors;
        if this median is consistently LOWER than neighbor_radius by at least
        epsilon, we are mostly scanning "non-neighbors" and can treat this
        as being in phase 2.

    Stop when:
      - we have at least K candidates, and
      - neighbor_radius and rm_window_median are available, and
      - neighbor_radius - rm_window_median >= epsilon.

    Expected state keys (in addition to K / num_candidates):
      - "neighbor_radius": Optional[float]
      - "rm_window_median": Optional[float]
      - "rm_window_size": int (optional)
      - "epsilon": float (optional tolerance, default ~1e-3)
    """
    K = state.get("K")
    num_candidates = state.get("num_candidates", 0)
    if K is None or num_candidates < K:
        return False, None

    neighbor_radius = state.get("neighbor_radius", None)
    rm_window_median = state.get("rm_window_median", None)
    epsilon: float = state.get("epsilon", 1e-3)

    # if we don't have the RM helpers, we cannot apply this policy
    if neighbor_radius is None or rm_window_median is None:
        return False, None

    # For similarity scores: neighbors have HIGH scores.
    # Once the median of recent probe scores is significantly LOWER
    # than the neighbor radius, we are effectively outside the "good"
    # neighborhood.
    if (neighbor_radius - rm_window_median) >= epsilon:
        return True, "rm_and_k"

    return False, None

# ---------------------------------------------------------------------------
# Policy selector
# ---------------------------------------------------------------------------

def get_early_stop_policy(name: Optional[str]):
    """
    Return a policy function by name.

    Known names:
      - None or "k_only"        -> stop_when_k_reached
      - "k_and_bound"           -> stop_when_k_and_bound
      - "k_and_stable"          -> stop_when_k_and_stable
      - "rm_and_k"              -> stop_when_rm_and_k
    """
    if name is None or name == "k_only":
        return stop_when_k_reached
    if name == "k_and_bound":
        return stop_when_k_and_bound
    if name == "k_and_stable":
        return stop_when_k_and_stable
    if name == "rm_and_k":
        return stop_when_rm_and_k
    # fallback
    return stop_when_k_reached