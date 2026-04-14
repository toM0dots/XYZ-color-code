# newman_moore_env.py
# Newman–Moore (triangular XOR) decoder environment with optional periodic (toric) boundary conditions.
# Spins live on vertices; checks live on downward triangles. Action = flip a spin.
# Reward: r = -ΔE, where ΔE = w_s*Δ|S| + λ_p*ΔL + λ_m*ΔD + (optional λ_b*ΔB on planar only).
# Frontier spins = those adjacent to k violated checks with k in allowed_k (default {2,3}).

from typing import Tuple, Dict, Any, List, Optional, Iterable
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("This module requires gymnasium. Install via `pip install gymnasium`.") from e


# ---------------- Utilities ----------------

def _tv_length_4n(S: np.ndarray, periodic: bool) -> int:
    """4-neighbor total variation (perimeter proxy). Adds wrap edges if periodic."""
    tv = np.abs(np.diff(S, axis=0)).sum() + np.abs(np.diff(S, axis=1)).sum()
    if periodic:
        tv += np.abs(S[0, :] - S[-1, :]).sum()
        tv += np.abs(S[:, 0] - S[:, -1]).sum()
    return int(tv)

def _boundary_dist_map(H: int, W: int) -> np.ndarray:
    """Manhattan distance to boundary on planar grid."""
    y = np.minimum(np.arange(H), np.arange(H)[::-1]).reshape(H, 1)
    x = np.minimum(np.arange(W), np.arange(W)[::-1]).reshape(1, W)
    return (x + y).astype(np.int32)

def _torus_delta(dx: int, L: int) -> int:
    dx = abs(dx)
    return min(dx, L - dx)

def _torus_manhattan(p, q, H, W) -> int:
    (i1, j1), (i2, j2) = p, q
    return _torus_delta(i1 - i2, H) + _torus_delta(j1 - j2, W)


# ---------------- Environment ----------------

class NewmanMooreDecoderEnv(gym.Env):
    """
    Boundary-first online decoder on the Newman–Moore (triangular XOR) lattice.

    Geometry:
      - Checks on an HxW grid; each check at (i,j) = s[i,j] XOR s[i+1,j] XOR s[i,j+1] (planar),
        or s ^ roll(s,-1,0) ^ roll(s,-1,1) (periodic).
      - Action flips a spin at vertex (r,c), toggling checks (r,c), (r-1,c), (r,c-1) with wrap if periodic.

    Observation:
      - Default: HxW binary syndrome S (1 = violated).
      - Optional: 3 channels [S, normalized boundary distance (planar only), local violated-degree].

    Reward shaping:
      ΔE = w_s*Δ|S| + λ_p*ΔL + λ_m*ΔD + λ_b*ΔB (λ_b ignored when periodic). Step reward r = -ΔE.

    Frontier:
      - Spins with k in allowed_k violated adjacent checks (default {2,3}); info["action_mask"] marks them.
      - If enforce_frontier=True, off-frontier actions are penalized (no state change).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        H: int = 16,
        W: int = 16,
        p_error: float = 0.06,
        periodic: bool = False,
        max_steps: Optional[int] = None,
        w_s: float = 1.0,
        lam_p: float = 0.2,
        lam_m: float = 0.05,
        lam_b: float = 0.0,   # ignored on periodic
        lam_c: float = 0.0,
        R_pair: int = 6,
        allowed_k: Iterable[int] = (2, 3),
        enforce_frontier: bool = False,
        obs_channels: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert H >= 1 and W >= 1
        self.H, self.W = H, W
        self.periodic = periodic
        self.p_error = p_error
        self.max_steps = max_steps if max_steps is not None else 6 * (H + W)
        self.w_s, self.lam_p, self.lam_m, self.lam_b, self.lam_c = w_s, lam_p, lam_m, (0.0 if periodic else lam_b), lam_c
        self.R_pair = R_pair
        self.allowed_k = set(int(k) for k in allowed_k)
        self.enforce_frontier = enforce_frontier
        self.obs_channels = obs_channels

        self._rng = np.random.default_rng(seed)

        # Spin grid size: (H,W) for periodic torus; (H+1,W+1) for planar
        if self.periodic:
            self.R, self.C = H, W
        else:
            self.R, self.C = H + 1, W + 1
        self.Nspin = self.R * self.C

        # Spaces
        if self.obs_channels == 1:
            self.observation_space = spaces.Box(low=0, high=1, shape=(H, W), dtype=np.int8)
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_channels, H, W), dtype=np.float32)
        self.action_space = spaces.Discrete(self.Nspin)

        # State
        self.spins = np.zeros((self.R, self.C), dtype=np.int8)
        self.S = np.zeros((H, W), dtype=np.int8)
        self._L = 0
        self._B = None      # boundary distance map (planar only)
        self.t = 0

    # ---------- geometry helpers ----------

    def _spin_index_to_rc(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.C)

    def _adjacent_checks(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Downward triangles touched by spin (r,c). Wrap modulo if periodic; clip if planar."""
        if self.periodic:
            H, W = self.H, self.W
            return [(r % H, c % W), ((r - 1) % H, c % W), ((r-1) % H, (c - 1) % W)]
        nbrs = []
        if 0 <= r < self.H and 0 <= c < self.W: nbrs.append((r, c))
        if 0 <  r <= self.H and 0 <= c < self.W: nbrs.append((r - 1, c))
        if 0 <= r < self.H and 0 <  c <= self.W: nbrs.append((r, c - 1))
        return nbrs

    def _compute_syndrome(self) -> np.ndarray:
        s = self.spins
        if self.periodic:
            # torus: checks also HxW, wrap neighbors
            S = (s ^ np.roll(s, -1, axis=0) ^ np.roll(np.roll(s, -1, axis=0), -1, axis = 1)).astype(np.int8)
            return S
        # planar: (H+1)x(W+1) spins → HxW checks
        return (s[:self.H, :self.W] ^
                s[1:self.H + 1, :self.W] ^
                s[:self.H, 1:self.W + 1]).astype(np.int8)

    def _frontier_mask(self, S: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.Nspin, dtype=bool)
        for idx in range(self.Nspin):
            r, c = self._spin_index_to_rc(idx)
            k = 0
            for (i, j) in self._adjacent_checks(r, c):
                k += S[i, j]
            if k in self.allowed_k:
                mask[idx] = True
        return mask

    # ---------- ΔE components ----------

    def _nearest_other_defect(self, S: np.ndarray, pos: Tuple[int, int]) -> int:
        """Nearest 1-site distance to another defect (torus or planar metric)."""
        if S[pos] == 0:
            return 0
        H, W = self.H, self.W
        coords = np.argwhere(S == 1)
        if coords.size == 0:
            return 0
        pi, pj = pos
        best = None
        for ii, jj in coords:
            if ii == pi and jj == pj:
                continue
            if self.periodic:
                d = _torus_manhattan((pi, pj), (ii, jj), H, W)
            else:
                d = abs(ii - pi) + abs(jj - pj)
            if best is None or d < best:
                best = d
                if best == 1:
                    break
        return 0 if best is None else int(best)

    def _delta_components(self, S: np.ndarray, spin_idx: int) -> Tuple[int, int, int, int]:
        """Compute (Δ|S|, ΔL, ΔD, ΔB) for flipping a spin."""
        r, c = self._spin_index_to_rc(spin_idx)
        adj = self._adjacent_checks(r, c)

        nS0 = int(S.sum())
        L0 = self._L

        # pre nearest-pair sum on affected checks
        pre_pts = [(i, j) for (i, j) in adj if S[i, j] == 1]
        D0 = sum(self._nearest_other_defect(S, (pi, pj)) for (pi, pj) in pre_pts)

        # Δ|S|
        n_viol = sum(S[i, j] for (i, j) in adj)
        nS1 = nS0 + (len(adj) - 2 * n_viol)
        dS = int(nS1 - nS0)

        # build S1 for ΔL/ΔD
        S1 = S.copy()
        for (i, j) in adj:
            S1[i, j] ^= 1

        L1 = _tv_length_4n(S1, self.periodic)
        dL = int(L1 - L0)

        post_pts = [(i, j) for (i, j) in adj if S1[i, j] == 1]
        D1 = sum(self._nearest_other_defect(S1, (pi, pj)) for (pi, pj) in post_pts)
        dD = int(D1 - D0)

        # boundary distance change only on planar
        if self.periodic or self.lam_b == 0.0:
            dB = 0
        else:
            if self._B is None:
                self._B = _boundary_dist_map(self.H, self.W)
            B0 = min(self._B[pi, pj] for (pi, pj) in pre_pts) if pre_pts else 0
            B1 = min(self._B[pi, pj] for (pi, pj) in post_pts) if post_pts else 0
            dB = int(B1 - B0)

        return dS, dL, dD, dB

    # ---------- state transition ----------

    def _apply_action(self, spin_idx: int):
        r, c = self._spin_index_to_rc(spin_idx)
        self.spins[r, c] ^= 1
        for (i, j) in self._adjacent_checks(r, c):
            self.S[i, j] ^= 1
        self._L = _tv_length_4n(self.S, self.periodic)

    # ---------- observation ----------

    def _obs(self) -> np.ndarray:
        if self.obs_channels == 1:
            return self.S.copy()
        # 3 channels: S, boundary distance (zeros on periodic), local violated-degree
        if not self.periodic and self._B is None:
            self._B = _boundary_dist_map(self.H, self.W)
        Bnorm = (self._B / (self._B.max() + 1e-8)) if (not self.periodic) else np.zeros_like(self.S, dtype=np.float32)

        S = self.S
        deg = np.zeros_like(S, dtype=np.float32)
        H, W = S.shape
        # 4-neighbor (wrap if periodic)
        if self.periodic:
            deg += np.roll(S, -1, axis=0)
            deg += np.roll(S, +1, axis=0)
            deg += np.roll(S, -1, axis=1)
            deg += np.roll(S, +1, axis=1)
        else:
            deg[:-1, :] += S[1:, :]
            deg[1:, :]  += S[:-1, :]
            deg[:, :-1] += S[:, 1:]
            deg[:, 1:]  += S[:, :-1]
        deg = deg / 4.0
        return np.stack([S.astype(np.float32), Bnorm.astype(np.float32), deg], axis=0)

    # ---------- Gym API ----------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.spins = (self._rng.random((self.R, self.C)) < self.p_error).astype(np.int8)
        self.S = self._compute_syndrome()
        self._L = _tv_length_4n(self.S, self.periodic)
        self._B = None
        self.t = 0
        return self._obs(), self._info()

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action index"
        mask = self._frontier_mask(self.S)
        invalid = self.enforce_frontier and not mask[action]

        if invalid:
            dS = dL = dD = dB = 0
            dE = 0.1   # small penalty
            reward = -float(dE)
        else:
            dS, dL, dD, dB = self._delta_components(self.S, action)
            dE = self.w_s * dS + self.lam_p * dL + self.lam_m * dD + self.lam_b * dB + self.lam_c * 0.0
            reward = -float(dE)
            self._apply_action(action)

        self.t += 1
        done = (self.S.sum() == 0)
        truncated = (self.t >= self.max_steps) and not done
        obs = self._obs()
        info = self._info(last_action=action, dS=int(dS), dL=int(dL), dD=int(dD), dB=int(dB), action_mask=mask)
        return obs, reward, done, truncated, info

    def _info(self, **extra) -> Dict[str, Any]:
        info = {"syndrome_count": int(self.S.sum()), "perimeter": int(self._L)}
        info.update(extra)
        info.setdefault("action_mask", self._frontier_mask(self.S))
        return info

    def render(self) -> str:
        return "\n".join(''.join('▲' if self.S[i, j] else '·' for j in range(self.W)) for i in range(self.H))

    # ---------- RL helpers ----------

    def frontier_actions(self) -> np.ndarray:
        """Indices of frontier spins (k in allowed_k)."""
        mask = self._frontier_mask(self.S)
        return np.flatnonzero(mask)

    def deltaE_for_actions(self, actions) -> np.ndarray:
        """ΔE for each candidate spin flip (no state change)."""
        dEs = []
        for a in actions:
            dS, dL, dD, dB = self._delta_components(self.S, int(a))
            dEs.append(float(self.w_s * dS + self.lam_p * dL + self.lam_m * dD + self.lam_b * dB + self.lam_c * 0.0))
        return np.asarray(dEs, dtype=np.float32)
