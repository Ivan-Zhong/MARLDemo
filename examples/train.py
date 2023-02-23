import argparse
from marldemo.runner.runner import Runner
import os
import os.path as osp
from pathlib import Path
import yaml

def get_args(algo, env):
    """Load config file for user-specified algo and env."""
    base_path = osp.split(osp.dirname(osp.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "marldemo", "configs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "marldemo", "configs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ippo",
        choices=[
            "ippo"
        ],
        help="Algorithm name. Choose from: ippo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "pettingzoo_mpe"
        ],
        help="Environment name. Choose from: pettingzoo_mpe.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    args, unparsed_args = parser.parse_known_args()

    algo_args, env_args = get_args(args.algo, args.env)

    # start training
    runner = Runner(args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
