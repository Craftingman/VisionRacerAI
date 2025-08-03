import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import sys
from torch.amp import GradScaler
from gymnasium.vector import AsyncVectorEnv
import os
import cv2

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    print("WARNING: CUDA not available; running on CPU.")

CHECKPOINT_PATH = "ppo_checkpoint.pt"

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                terminated = term
                truncated = trunc
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class InsaneCarRacingWrapper(gym.Wrapper):
    def __init__(self, env, branch_prob=0.5, perturb_prob=0.02, perturb_duration=5):
        super().__init__(env)
        self.branch_prob = branch_prob
        self.perturb_prob = perturb_prob
        self.perturb_duration = perturb_duration
        self.reset_internal()

    def reset_internal(self):
        self.perturb_steps_left = 0
        self.perturb_strength = 0.0
        self.steps = 0
        self.forked = False
        self.do_branch = False
        self.fork_direction = 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_internal()
        self.do_branch = random.random() < self.branch_prob
        return obs, info

    def step(self, action):
        self.steps += 1
        action = action.copy()

        if self.perturb_steps_left > 0:
            action[0] = np.clip(action[0] + self.perturb_strength, -1.0, 1.0)
            self.perturb_steps_left -= 1
        elif random.random() < self.perturb_prob:
            self.perturb_steps_left = self.perturb_duration
            self.perturb_strength = random.uniform(-0.5, 0.5)

        if self.do_branch and not self.forked and 50 < self.steps < 120:
            self.fork_direction = random.choice([-1, 1])
            self.forked = True

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if self.forked and 50 < self.steps < 150:
            steer = action[0]
            reward *= 1.05 if np.sign(steer) == self.fork_direction or abs(steer) <= 0.1 else 0.85

        return obs, reward, terminated, truncated, info

def make_env():
    def _thunk():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = InsaneCarRacingWrapper(env)
        return env
    return _thunk

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = x / 255.0
        features = self.backbone(x)
        mu = torch.tanh(self.mu_head(features))
        std = torch.exp(self.log_std)
        value = self.value_head(features).squeeze(-1)
        return mu, std, value

def compute_gae_vec(rewards, values, masks, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    advantages = torch.zeros((T, N), device=rewards.device)
    gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        gae = delta + gamma * lam * masks[t] * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

def ppo_update(policy, optimizer, obs, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, scaler=None):
    obs = obs.detach()
    actions = actions.detach()
    old_log_probs = old_log_probs.detach()
    returns = returns.detach()
    advantages = advantages.detach()

    with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
        mu, std, values = policy(obs)
        dist = Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    return policy_loss.item(), value_loss.item(), entropy.item()

def try_compile(model):
    if sys.platform == "win32" or not hasattr(torch, "compile"):
        return model
    try:
        import triton
        return torch.compile(model)
    except Exception:
        return model

def show_obs_grid(obs_batch, num_cols=4, scale=3):
    # obs_batch: (N, H, W, C) uint8
    N, H, W, C = obs_batch.shape
    num_display = min(N, num_cols * ((N + num_cols - 1) // num_cols))
    rows = (num_display + num_cols - 1) // num_cols
    grid_h = rows * H
    grid_w = num_cols * W
    grid = np.zeros((grid_h, grid_w, C), dtype=np.uint8)
    for idx in range(num_display):
        r = idx // num_cols
        c = idx % num_cols
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = obs_batch[idx]

    scaled = cv2.resize(grid, (grid_w * scale, grid_h * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Training observations (first envs)", cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR))

def test_episode(env, policy, device, max_steps=1000, render=False, stochastic=False):
    total_reward = 0.0
    num_episodes = 1

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
        ep_reward = 0.0

        for _ in range(max_steps):
            with torch.no_grad():
                mu, std, _ = policy(obs_tensor)
                if stochastic:
                    dist = Normal(mu, std)
                    action = dist.sample()
                else:
                    action = mu
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += reward
            if render:
                env.render()
            if terminated or truncated:
                break
            obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)

        total_reward += ep_reward
    avg_reward = total_reward / num_episodes
    print(f"[Test] Average Return over {num_episodes} episodes ({'stochastic' if stochastic else 'deterministic'}): {avg_reward:.2f}")

def train():
    num_envs = 8
    train_envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    eval_env = InsaneCarRacingWrapper(gym.make("CarRacing-v3", render_mode="human"))

    obs_shape = (3, 96, 96)
    action_dim = 3
    policy = try_compile(ActorCritic(obs_shape, action_dim).to(device))
    optimizer = optim.Adam(policy.parameters(), lr=2.5e-4)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    start_epoch = 1

    if os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    max_epochs = 300
    steps_per_epoch = 400
    update_epochs = 4
    minibatch_size = 256
    gamma, lam = 0.99, 0.95

    vis_interval = 5 # How often do we update the visualization 
    for epoch in range(start_epoch, max_epochs + 1):
        obs, _ = train_envs.reset()
        obs_tensor = torch.from_numpy(obs).permute(0, 3, 1, 2).float().to(device)

        obs_buf, actions_buf, logp_buf, value_buf = [], [], [], []
        reward_buf, mask_buf = [], []
        ep_returns = np.zeros(num_envs, dtype=np.float32)
        ep_lengths = np.zeros(num_envs, dtype=np.int32)
        finished_returns = []

        for step in range(steps_per_epoch):
            mu, std, value = policy(obs_tensor)
            with torch.no_grad():
                print(f"[Debug] μ mean: {mu.mean().item():.4f}, std: {std.mean().item():.4f}")
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action_clipped = torch.clamp(action, -1.0, 1.0)
            cpu_actions = action_clipped.cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = train_envs.step(cpu_actions)
            dones = np.logical_or(terminated, truncated)
            masks = 1.0 - dones.astype(np.float32)

            # Visualisation itself
            if step % vis_interval == 0:
                try:
                    show_obs_grid(next_obs[:8], num_cols=4)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Visualization closed by user.")
                        cv2.destroyAllWindows()
                except Exception as e:
                    print(f"[Viz error] {e}")

            # Track episode stats
            for i in range(num_envs):
                ep_returns[i] += rewards[i]
                ep_lengths[i] += 1
                if dones[i]:
                    finished_returns.append(ep_returns[i])
                    ep_returns[i] = 0
                    ep_lengths[i] = 0

            obs_buf.append(obs_tensor.cpu())
            actions_buf.append(action.cpu())
            logp_buf.append(log_prob.cpu())
            value_buf.append(value.cpu())
            reward_buf.append(torch.tensor(rewards, dtype=torch.float32))
            mask_buf.append(torch.tensor(masks, dtype=torch.float32))

            obs = next_obs
            obs_tensor = torch.from_numpy(obs).permute(0, 3, 1, 2).float().to(device)

        with torch.no_grad():
            _, _, last_value = policy(obs_tensor)
            value_buf.append(last_value.cpu())

        obs_tensor_batch = torch.stack(obs_buf)
        actions_tensor_batch = torch.stack(actions_buf)
        logp_tensor_batch = torch.stack(logp_buf)
        value_tensor_batch = torch.stack(value_buf)
        reward_tensor_batch = torch.stack(reward_buf)
        mask_tensor_batch = torch.stack(mask_buf)

        advantages, returns = compute_gae_vec(
            reward_tensor_batch.to(device),
            value_tensor_batch.to(device),
            mask_tensor_batch.to(device),
            gamma=gamma, lam=lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T, N = advantages.shape
        batch_size = T * N
        obs_flat = obs_tensor_batch.reshape(batch_size, *obs_shape).to(device)
        actions_flat = actions_tensor_batch.reshape(batch_size, action_dim).to(device)
        old_logp_flat = logp_tensor_batch.reshape(batch_size).to(device)
        returns_flat = returns.reshape(batch_size).to(device)
        adv_flat = advantages.reshape(batch_size).to(device)

        indices = torch.randperm(batch_size)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(update_epochs):
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                p_loss, v_loss, entropy = ppo_update(
                    policy, optimizer,
                    obs_flat[mb_idx], actions_flat[mb_idx],
                    old_logp_flat[mb_idx],
                    returns_flat[mb_idx], adv_flat[mb_idx],
                    scaler=scaler
                )
                policy_losses.append(p_loss)
                value_losses.append(v_loss)
                entropies.append(entropy)

        avg_return = np.mean(finished_returns) if finished_returns else 0.0
        print(f"[Epoch {epoch:03}] Avg Return: {avg_return:.1f} | "
              f"Policy Loss: {np.mean(policy_losses):.4f} | "
              f"Value Loss: {np.mean(value_losses):.4f} | "
              f"Entropy: {np.mean(entropies):.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None
        }, CHECKPOINT_PATH)

        if epoch % 2 == 0:
            test_episode(eval_env, policy, device, render=True, stochastic=True)

    train_envs.close()
    eval_env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train()