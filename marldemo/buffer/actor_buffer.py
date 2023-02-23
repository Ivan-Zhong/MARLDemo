import torch
import numpy as np
from marldemo.util.util import get_shape_from_act_space

class ActorBuffer:
    """
    ActorBuffer contains data for on-policy actors.
    """
    def __init__(self, args, obs_space, act_space):
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_shape = obs_space.shape

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)

        self.step = 0

    def insert(self, obs, actions, action_log_probs):
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()

    def feed_forward_generator_actor(self, advantages, actor_num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= actor_num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          actor_num_mini_batch))
            mini_batch_size = batch_size // actor_num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size]
                   for i in range(actor_num_mini_batch)]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield obs_batch, actions_batch, old_action_log_probs_batch, adv_targ