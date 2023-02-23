import torch.nn as nn
from marldemo.util.util import init, get_init_method

"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method):
        super(MLPLayer, self).__init__()

        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain("relu")

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [init_(nn.Linear(input_dim, hidden_sizes[0])), nn.ReLU(), nn.LayerNorm(hidden_sizes[0])]

        for i in range(1, len(hidden_sizes)):
            layers += [init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])), nn.ReLU(), nn.LayerNorm(hidden_sizes[i])]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self.initialization_method = args["initialization_method"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_dim = obs_shape[0]

        self.feature_norm = nn.LayerNorm(obs_dim)
        self.mlp = MLPLayer(obs_dim, self.hidden_sizes, self.initialization_method)

    def forward(self, x):
        x = self.feature_norm(x)
        x = self.mlp(x)

        return x
