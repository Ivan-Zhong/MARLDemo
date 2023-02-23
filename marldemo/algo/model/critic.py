import torch
import torch.nn as nn
from marldemo.util.util import init, check
from marldemo.algo.model.mlp import MLPBase
from marldemo.util.util import get_init_method


class Critic(nn.Module):
    """
    Critic network class.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = cent_obs_space.shape
        self.base = MLPBase(args, cent_obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.

        :return values: (torch.Tensor) value function predictions.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values
