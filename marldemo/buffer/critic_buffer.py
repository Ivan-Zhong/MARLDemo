import torch
import numpy as np


class CriticBuffer:
    def __init__(self, args, share_obs_space):
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.gamma = args["gamma"]
        self.gae_lambda = args["gae_lambda"]

        share_obs_shape = share_obs_space.shape

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape), dtype=np.float32)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        
        self.bad_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, value_preds, rewards, masks, bad_masks):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_rewards(self):
        return np.mean(self.rewards)

    def compute_returns(self, next_value):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step] + self.gamma * self.value_preds[step +
                                                                    1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + self.gamma * \
                self.gae_lambda * self.masks[step + 1] * gae
            gae = self.bad_masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator_critic(self, critic_num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= critic_num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          critic_num_mini_batch))
            mini_batch_size = batch_size // critic_num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size]
                   for i in range(critic_num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]

            yield share_obs_batch, value_preds_batch, return_batch