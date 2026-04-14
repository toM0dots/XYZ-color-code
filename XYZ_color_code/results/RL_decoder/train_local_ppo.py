# train_local_ppo.py
# pip install torch gymnasium numpy
import os, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from local_patch_env import LocalPatchEnv

def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

class LocalPolicy(nn.Module):
    """Takes (B, 3, 3) syndrome patch → logits(2), value(1)."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=0), nn.ReLU(),   # 3x3 → 1x1
        )
        self.pi = nn.Linear(32, 2)
        self.v  = nn.Linear(32, 1)

    def forward(self, obs):                 # obs: (B, 3, 3)
        x = obs.unsqueeze(1)                # (B,1,3,3)
        f = self.body(x).view(x.size(0), -1)  # (B, 32)
        return self.pi(f), self.v(f).squeeze(-1)

class VecLocal:
    """Vectorized runner over N LocalPatchEnv instances."""
    def __init__(self, n, H, W, p_error, max_steps, seed):
        base_seed = seed if seed is not None else int(time.time())
        self.envs = [LocalPatchEnv(H=H, W=W, p_error=p_error, max_steps=max_steps, seed=base_seed+i)
                     for i in range(n)]
        self.n = n

    def reset(self):
        obs, infos = [], []
        for e in self.envs:
            o, info = e.reset()
            obs.append(o); infos.append(info)
        return np.stack(obs, 0), infos

    def step(self, actions):
        obs, rewards, dones, truncs, infos = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, t, info = e.step(int(a))
            obs.append(o); rewards.append(r); dones.append(d); truncs.append(t); infos.append(info)
        return np.stack(obs, 0), np.array(rewards, float), np.array(dones, bool), np.array(truncs, bool), infos

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    vec = VecLocal(n=args.num_envs, H=args.H, W=args.W, p_error=args.p, max_steps=args.max_steps, seed=args.seed)
    policy = LocalPolicy().to(device)
    optim  = torch.optim.Adam(policy.parameters(), lr=args.lr)

    B = vec.n
    steps_per_update = args.rollout_len * B

    obs_np, _ = vec.reset()
    obs = to_t(obs_np, device)

    for upd in range(1, args.updates+1):
        # buffers
        obs_buf   = torch.zeros((steps_per_update, B, 3, 3), device=device)
        act_buf   = torch.zeros((steps_per_update, B), dtype=torch.long, device=device)
        logp_buf  = torch.zeros((steps_per_update, B), device=device)
        val_buf   = torch.zeros((steps_per_update, B), device=device)
        rew_buf   = torch.zeros((steps_per_update, B), device=device)
        done_buf  = torch.zeros((steps_per_update, B), dtype=torch.bool, device=device)

        t_idx = 0
        for _ in range(args.rollout_len):
            logits, v = policy(obs)              # (B,2), (B,)
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)

            obs2_np, r_np, d_np, t_np, _ = vec.step(a.detach().cpu().numpy())
            r = to_t(r_np, device)
            d = torch.from_numpy(d_np | t_np).to(device)

            # store (detach rollout tensors)
            obs_buf[t_idx]   = obs
            act_buf[t_idx]   = a
            logp_buf[t_idx]  = logp.detach()
            val_buf[t_idx]   = v.detach()
            rew_buf[t_idx]   = r.detach()
            done_buf[t_idx]  = d

            obs = to_t(obs2_np, device)
            t_idx += 1

        # bootstrap & GAE (no grad)
        with torch.no_grad():
            _, v_last = policy(obs)
            val_det  = val_buf
            rew_det  = rew_buf
            done_det = done_buf

            adv = torch.zeros_like(val_det)
            gae = torch.zeros(B, device=device)
            for t in reversed(range(steps_per_update)):
                nonterm = (~done_det[t]).float()
                next_value = v_last if t == steps_per_update - 1 else val_det[t+1]
                delta = rew_det[t] + args.gamma * next_value * nonterm - val_det[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + val_det

        # flatten
        obs_f   = obs_buf.reshape(-1, 3, 3)
        act_f   = act_buf.reshape(-1)
        oldlogp = logp_buf.reshape(-1)
        adv_f   = adv.reshape(-1)
        ret_f   = ret.reshape(-1)

        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        # PPO updates
        batch = obs_f.size(0)
        idxs  = torch.arange(batch, device=device)
        for _ in range(args.epochs):
            perm = idxs[torch.randperm(batch)]
            for start in range(0, batch, args.minibatch):
                mb = perm[start:start+args.minibatch]
                logits, v = policy(obs_f[mb])
                dist = Categorical(logits=logits)
                newlogp = dist.log_prob(act_f[mb])

                ratio = (newlogp - oldlogp[mb]).exp()
                surr1 = ratio * adv_f[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_f[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (v - ret_f[mb]).pow(2).mean()
                entropy_loss = -args.ent_coef * dist.entropy().mean()

                loss = policy_loss + value_loss + entropy_loss
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()

        if upd % args.log_interval == 0:
            # quick stat: average |S| over envs
            avg_S = np.mean([e.base.S.sum() for e in vec.envs])
            print(f"[{upd:04d}] avg|S|={avg_S:.2f}")

        if upd % args.save_interval == 0:
            path = os.path.join(args.out, f"localppo_H{vec.envs[0].H}_W{vec.envs[0].W}_upd{upd}.pt")
            torch.save({"model": policy.state_dict()}, path)
            print("saved", path)

    path = os.path.join(args.out, f"localppo_H{vec.envs[0].H}_W{vec.envs[0].W}_final.pt")
    torch.save({"model": policy.state_dict()}, path)
    print("saved", path)

def parse():
    pa = argparse.ArgumentParser()
    pa.add_argument('--H', type=int, default=16)
    pa.add_argument('--W', type=int, default=16)
    pa.add_argument('--p', type=float, default=0.06)
    pa.add_argument('--max_steps', type=int, default=None)

    pa.add_argument('--num_envs', type=int, default=32)
    pa.add_argument('--rollout_len', type=int, default=128)
    pa.add_argument('--updates', type=int, default=400)

    pa.add_argument('--gamma', type=float, default=0.99)
    pa.add_argument('--lam', type=float, default=0.95)
    pa.add_argument('--clip', type=float, default=0.2)
    pa.add_argument('--ent_coef', type=float, default=0.01)
    pa.add_argument('--lr', type=float, default=3e-4)
    pa.add_argument('--minibatch', type=int, default=256)
    pa.add_argument('--epochs', type=int, default=4)

    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--out', type=str, default='checkpoints_local')
    pa.add_argument('--save_interval', type=int, default=50)
    pa.add_argument('--log_interval', type=int, default=10)
    pa.add_argument('--cpu', action='store_true')
    return pa.parse_args()

if __name__ == '__main__':
    args = parse()
    train(args)
