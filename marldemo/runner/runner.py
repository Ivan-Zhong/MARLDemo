
import time
import numpy as np
import torch
from marldemo.buffer.actor_buffer import ActorBuffer
from marldemo.buffer.critic_buffer import CriticBuffer
from marldemo.algo.ippo_actor import IPPOActor
from marldemo.algo.ippo_critic import IPPOCritic
import time
import numpy as np
import torch
from marldemo.util.util import _t2n
import setproctitle
from marldemo.util.util import make_eval_env, make_train_env, make_render_env, seed, init_device, init_dir, save_config
from marldemo.env.pettingzoo_mpe.logger import PettingZooMPELogger


class Runner:
    """Runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the Runner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        # get practical parameters
        for k, v in algo_args["train"].items():
            self.__dict__[k] = v
        for k, v in algo_args["eval"].items():
            self.__dict__[k] = v
        for k, v in algo_args["render"].items():
            self.__dict__[k] = v
        # TODO: seed --> set_seed
        seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.use_render:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args.env, env_args, args.algo, args.exp_name, algo_args["seed"]["seed"])
            save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args.algo) + "-" + str(args.env) + "-" + str(args.exp_name))

        # set the config of env
        if self.use_render:
            self.envs = make_render_env(algo_args["seed"]["seed"], env_args)
        else:            
            self.envs = make_train_env(
                algo_args["seed"]["seed"], 
                algo_args["train"]["n_rollout_threads"], 
                env_args
                )
            self.eval_envs = make_eval_env(
                algo_args["seed"]["seed"], 
                algo_args["eval"]["n_eval_rollout_threads"], 
                env_args
                ) if algo_args["eval"]["use_eval"] else None
        self.num_agents = self.envs.n_agents

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        # actor
        self.actor = []
        for agent_id in range(self.num_agents):
            ac = IPPOActor(
                {**algo_args["model"], **algo_args["algo"]}, self.envs.observation_space[agent_id], self.envs.action_space[agent_id], device=self.device)
            self.actor.append(ac)

        if self.use_render is False:
            # Buffer for rendering
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = ActorBuffer(
                    {**algo_args["train"], **algo_args["model"]}, self.envs.observation_space[agent_id], self.envs.action_space[agent_id])
                self.actor_buffer.append(ac_bu)

            share_observation_space = self.envs.share_observation_space[0]
            self.critic = IPPOCritic(
                {**algo_args["model"], **algo_args["algo"]}, share_observation_space, device=self.device)
            self.critic_buffer = CriticBuffer(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]}, share_observation_space)

            self.logger = PettingZooMPELogger(
            args, algo_args, env_args, self.num_agents, self.writter, self.run_dir)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        if self.use_render is True:
            self.render()
            return
        print("start running")
        self.warmup()

        episodes = int(
            self.num_env_steps) // self.episode_length // self.n_rollout_threads

        self.logger.init(episodes)

        for episode in range(1, episodes + 1):

            self.logger.episode_init(episode)

            self.prep_rollout()
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs = self.collect(
                    step)
                # actions: (n_threads, n_agents, action_dim)
                obs, share_obs, rewards, dones, infos = self.envs.step(
                    actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                data = obs, share_obs, rewards, dones, infos, \
                    values, actions, action_log_probs

                self.logger.per_step(data)

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            self.prep_training()
            actor_train_infos, critic_train_info = self.train()

            # log information
            if episode % self.log_interval == 0:
                self.logger.episode_log(
                    actor_train_infos, critic_train_info, self.actor_buffer, self.critic_buffer)

            # eval
            if episode % self.eval_interval == 0:
                if self.use_eval:
                    self.prep_rollout()
                    self.eval()
                else:
                    self.save()

            self.after_update()

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
        self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()

    @torch.no_grad()
    def collect(self, step):
        action_collector = []
        action_log_prob_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob = self.actor[agent_id].get_actions(self.actor_buffer[agent_id].obs[step])
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
        # [self.envs, agents, dim]
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(
            action_log_prob_collector).transpose(1, 0, 2)


        value = self.critic.get_values(self.critic_buffer.share_obs[step])
        values = _t2n(value)     

        return values, actions, action_log_probs

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
            values, actions, action_log_probs = data

        dones_env = np.all(dones, axis=1)

        # masks use 0 to mask out threads that just finish
        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        
        # bad_masks use 0 to denote truncation and 1 to denote termination
        bad_masks = np.array([[0.0] if "bad_transition" in info[0].keys() and info[0]["bad_transition"] == True else [1.0] for info in infos])


        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(obs[:, agent_id], actions[:, agent_id],
                                               action_log_probs[:, agent_id])
        self.critic_buffer.insert(
            share_obs[:, 0], values, rewards[:, 0], masks[:, 0], bad_masks)
          

    @torch.no_grad()
    def compute(self):
        next_value = self.critic.get_values(self.critic_buffer.share_obs[-1])
        next_value = _t2n(next_value)
        self.critic_buffer.compute_returns(next_value)

    def train(self):
        actor_train_infos = []
        advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
        for agent_id in range(self.num_agents):
            actor_train_info = self.actor[agent_id].train(self.actor_buffer[agent_id], advantages.copy())
            actor_train_infos.append(actor_train_info)
        critic_train_info = self.critic.train(self.critic_buffer)
        return actor_train_infos, critic_train_info

    def after_update(self):
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0
        eval_obs, eval_share_obs = self.eval_envs.reset()
        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions= \
                    self.actor[agent_id].act(eval_obs[:, agent_id], deterministic=True)
                eval_actions_collector.append(_t2n(eval_actions))
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions)
            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos
            self.logger.eval_per_step(eval_data)
            eval_dones_env = np.all(eval_dones, axis=1)
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)
            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                save_model = self.logger.eval_log(eval_episode)
                if save_model:
                    self.save()
                break

    @torch.no_grad()
    def render(self):
        print("start rendering")
        for _ in range(self.render_episodes):
            eval_obs, _ = self.envs.reset()
            eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
            rewards = 0
            while True:
                eval_actions_collector = []
                for agent_id in range(self.num_agents):
                    eval_actions = \
                        self.actor[agent_id].act(eval_obs[:, agent_id],
                                                deterministic=True)
                    eval_actions_collector.append(_t2n(eval_actions))
                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                # Obser reward and next obs
                eval_obs, _, eval_rewards, eval_dones, _ = self.envs.step(
                    eval_actions[0])
                rewards += eval_rewards[0][0]
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                self.envs.render()
                time.sleep(0.1)
                if eval_dones[0]:
                    print(f"total reward of this episode: {rewards}")
                    break

    def prep_rollout(self):
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(policy_actor.state_dict(), str(
                self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
        policy_critic = self.critic.critic
        torch.save(policy_critic.state_dict(), str(
            self.save_dir) + "/critic_agent" + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.actor[agent_id].actor.load_state_dict(
                policy_actor_state_dict)
        if not self.use_render:
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + '/critic_agent' + '.pt')
            self.critic.critic.load_state_dict(
                policy_critic_state_dict)

    def close(self):
        # post process
        if self.use_render:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(
                str(self.log_dir + '/summary.json'))
            self.writter.close()
            self.logger.close()
