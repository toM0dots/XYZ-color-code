# local_patch_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from newman_moore_env import NewmanMooreDecoderEnv  # your base env (periodic-capable)

class LocalPatchEnv(gym.Env):
    """
    Local training wrapper:
      - Underlying lattice is NewmanMooreDecoderEnv (MUST be periodic=True).
      - Each step chooses a random *spin* (center vertex).
      - Observation is a 3x3 window of the *syndrome grid* S around that center (with wrap).
      - Action space: Discrete(2) => 0 = do nothing, 1 = flip the *center spin only*.
      - Reward: r = -Δ|S| (the only term). Done when |S| == 0 or max_steps.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self,
                 H: int = 16,
                 W: int = 16,
                 p_error: float = 0.06,
                 max_steps: Optional[int] = None,
                 seed: Optional[int] = None):
        super().__init__()
        # base env on a torus
        self.base = NewmanMooreDecoderEnv(H=H, W=W, p_error=p_error,
                                          periodic=True, max_steps=max_steps,
                                          # all shaping off—reward is provided here
                                          w_s=0.0, lam_p=0.0, lam_m=0.0, lam_b=0.0)
        self.H, self.W = H, W
        self.max_steps = self.base.max_steps
        self._rng = np.random.default_rng(seed)
        # Local observation: 3x3 patch of S
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.int8)
        # Binary action: 0 stay, 1 flip center spin
        self.action_space = spaces.Discrete(2)

        # state for current local decision
        self.center_idx: Optional[int] = None
        self.t = 0

    # ---------- helpers ----------
    def _center_rc(self) -> Tuple[int, int]:
        return divmod(self.center_idx, self.base.C)

    def _syndrome_patch_3x3(self) -> np.ndarray:
        """3x3 window of the base syndrome around the current *spin* center."""
        # We map the spin (r,c) to nearby checks by simply using (r + di, c + dj) on the HxW S grid (wrap)
        S = self.base.S
        r, c = self._center_rc()
        H, W = self.H, self.W
        patch = np.empty((3,3), dtype=np.int8)
        # offsets -1,0,1 in both directions with torus wrap
        offs = (-1, 0, 1)
        for i, di in enumerate(offs):
            for j, dj in enumerate(offs):
                patch[i, j] = S[(r + di) % H, (c + dj) % W]
        return patch

    def _new_center(self):
        self.center_idx = int(self._rng.integers(self.base.Nspin))

    # ---------- Gym API ----------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _, _ = self.base.reset(seed=seed)  # discard base's obs; we build local obs
        self.t = 0
        self._new_center()
        obs = self._syndrome_patch_3x3()
        info = self._info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action"
        S0 = int(self.base.S.sum())
        # apply flip only if action==1
        if action == 1:
            # flip the *center* spin (no penalties or other shaping)
            self.base._apply_action(self.center_idx)
        S1 = int(self.base.S.sum())
        # local reward: only total syndrome count change
        r = float(-(S1 - S0))  # = -Δ|S|

        self.t += 1
        done = (S1 == 0)
        trunc = (self.t >= self.max_steps) and not done

        # pick a new random center for the *next* local decision
        self._new_center()
        obs = self._syndrome_patch_3x3()
        info = self._info(dS=(S1-S0))
        return obs, r, done, trunc, info

    def _info(self, **extra):
        info = {
            "total_syndrome": int(self.base.S.sum()),
            "center_idx": self.center_idx,
        }
        info.update(extra)
        return info

    def render(self) -> str:
        return self.base.render()
