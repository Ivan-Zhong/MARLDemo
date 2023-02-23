import numpy as np
import torch
import torch.nn as nn
from marldemo.util.util import get_grad_norm, check
from marldemo.algo.model.actor import Actor


class IPPOActor:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize IPPO algorithm."""
# save arguments
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.lr = args["lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        # save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space
        # create actor network
        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        # create actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

    def get_actions(self, obs, deterministic=False):
        """Compute actions and value function predictions for the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, action_log_probs = self.actor(obs, deterministic)
        return actions, action_log_probs

    def evaluate_actions(self, obs, action):
        """Get action logprobs / entropy and value function predictions for actor update.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            action: (np.ndarray) actions whose log probabilites and entropy to compute.
        """

        action_log_probs, dist_entropy, action_distribution = self.actor.evaluate_actions(obs, action)
        return action_log_probs, dist_entropy, action_distribution

    def act(self, obs, deterministic=False):
        """Compute actions using the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _ = self.actor(obs, deterministic)
        return actions

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        (
            obs_batch,
            actions_batch,
            old_action_log_probs_batch,
            adv_targ
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            actions_batch,
        )
        # update actor
        imp_weights = torch.prod(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (ActorBuffer) buffer containing training data related to actor.
            advantages: (ndarray) advantages.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.actor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.actor.eval()