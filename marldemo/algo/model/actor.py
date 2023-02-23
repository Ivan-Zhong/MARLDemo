import torch
import torch.nn as nn
from marldemo.util.util import init, check
from marldemo.algo.model.mlp import MLPBase
from marldemo.algo.model.act import ACTLayer


class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = obs_space.shape
        self.base = MLPBase(args, obs_shape)

        self.act = ACTLayer(action_space, self.hidden_sizes[-1],
                            self.initialization_method, self.gain, args)

        self.to(device)

    def forward(self, obs, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        obs = check(obs).to(**self.tpdv)

        actor_features = self.base(obs)

        actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs

    def evaluate_actions(self, obs, action):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        actor_features = self.base(obs)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(actor_features, action)

        return action_log_probs, dist_entropy, action_distribution
