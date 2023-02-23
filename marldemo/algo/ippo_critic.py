import numpy as np
import torch
import torch.nn as nn
from marldemo.util.util import get_grad_norm, huber_loss, mse_loss
from marldemo.util.util import check
from marldemo.algo.model.critic import Critic


class IPPOCritic:
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.clip_param = args["clip_param"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]

        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.share_obs_space = cent_obs_space

        self.critic = Critic(args, self.share_obs_space, self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def get_values(self, cent_obs):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.

        :return values: (torch.Tensor) value function predictions.
        """
        value = self.critic(cent_obs)
        return value

    def cal_value_loss(self, values, value_preds_batch, return_batch,):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def update(self, sample):
        """
        Update critic network.
        :param sample: (Tuple) contains data batch with which to update networks.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, value_preds_batch, return_batch = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values = self.get_values(share_obs_batch)

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch)

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train(self, critic_buffer):
        """
        Perform a training update using minibatch GD.
        :param critic_buffer: (CriticBuffer) buffer containing training data related to critic.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info['value_loss'] = 0
        train_info['critic_grad_norm'] = 0

        for _ in range(self.critic_epoch):
            data_generator = critic_buffer.feed_forward_generator_critic(
                    self.critic_num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm = self.update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['critic_grad_norm'] += critic_grad_norm

        num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.critic.train()

    def prep_rollout(self):
        self.critic.eval()
