import ray
import time
import torch
import numpy as np

from Parameters import *
from collections import OrderedDict
from torch.distributions.categorical import Categorical
from Utils import set_env, set_network, convert_to_tensor, discount


@ray.remote(num_cpus=1, num_gpus=0)
class Runner(object):
    """
    Actor object that runs the simulation and collects experience.
    """

    def __init__(self, meta_agent_id):
        self.id = meta_agent_id
        self.device = torch.device('cpu')
        self.curr_episode = int(meta_agent_id)

        self.model_type = NETWORK_PARAMS.NET_TYPE
        self.env = set_env(env_type=INPUT_PARAMS.ENV_TYPE,
                           server_number=meta_agent_id)
        self.input_size = self.env.obs_space_n
        self.output_size = self.env.action_space_n
        self.agent_size = self.env.agent_space_n
        self.local_network = set_network(input_size=self.env.obs_space_n,
                                         output_size=self.env.action_space_n,
                                         agent_size=self.env.agent_space_n,
                                         device=self.device)

        self.experience_buffers = None
        self.returned_buffers = None
        self.bootstrap_values = None
        self.rnn_states_actor = None
        self.rnn_states_critic = None

        self.episode_step = None
        self.action_change = None
        self.episode_reward = None
        self.gif_episode = None
        self.episode_eval_metrics = None

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def initialization(self):
        # Initialize variables
        self.episode_step = 0
        self.action_change = 0
        self.episode_reward = 0
        self.gif_episode = int(self.curr_episode)
        self.episode_eval_metrics = OrderedDict(avg_queue=[], std_queue=[], avg_speed=[], avg_travel=[])

        # Initialize dicts
        self.experience_buffers = [[] for _ in range(8)]
        self.returned_buffers = [[] for _ in range(8)]
        self.bootstrap_values = None
        self.rnn_states_actor = None
        self.rnn_states_critic = [None, None]

    def run_episode_single_threaded(self):
        # Initialize variables and buffer
        start_time = time.time()
        self.initialization()
        observations = self.env.reset()
        while True:
            action_dict = OrderedDict()

            multi_agent_obs = np.stack(list(observations.values()))
            multi_agent_obs = convert_to_tensor(data=multi_agent_obs.copy(),
                                                data_type=torch.float32,
                                                device=self.device)
            multi_agent_mask_n = self.env.calculate_neighbor_mask()
            multi_agent_mask = np.stack(list(multi_agent_mask_n.values()))
            multi_agent_mask = convert_to_tensor(multi_agent_mask.copy(),
                                                 data_type=torch.float32,
                                                 device=self.device)

            with torch.no_grad():
                multi_agent_policy, _, self.rnn_states_actor = self.local_network.forward(multi_agent_obs,
                                                                                          self.rnn_states_actor,
                                                                                          multi_agent_mask)
            multi_agent_pi_dist = Categorical(multi_agent_policy)
            multi_agent_actions = multi_agent_pi_dist.sample()
            multi_agent_log_p = multi_agent_pi_dist.log_prob(multi_agent_actions)

            action_list = multi_agent_actions.numpy().reshape(-1).tolist()
            for i, action in enumerate(action_list):
                action_dict[self.env.rl_agent_id[i]] = action

            neighbor_actions_dict = self.env.get_neighbor_actions(action_dict=action_dict)
            multi_agent_neighbor_actions = np.array(list(neighbor_actions_dict.values()))
            multi_agent_neighbor_actions = convert_to_tensor(data=multi_agent_neighbor_actions.copy(),
                                                             data_type=torch.int64,
                                                             device=self.device)

            next_observations, rewards, done, info = self.env.step(action_dict=action_dict)

            multi_agent_target_queue_vec = np.array(list(info[3].values()))
            multi_agent_target_queue_vec = convert_to_tensor(data=multi_agent_target_queue_vec.copy(),
                                                             data_type=torch.float32,
                                                             device=self.device)
            multi_agent_neighbor_rewards = np.array(list(info[4].values()))

            # Store trajectory data
            self.experience_buffers[0].append(multi_agent_obs)
            self.experience_buffers[1].append(multi_agent_actions.reshape(-1))
            self.experience_buffers[2].append(multi_agent_neighbor_rewards)
            self.experience_buffers[3].append(multi_agent_log_p.reshape(-1))
            self.experience_buffers[4].append(multi_agent_neighbor_actions)
            self.experience_buffers[5].append(multi_agent_policy[0])
            self.experience_buffers[6].append(multi_agent_target_queue_vec)
            self.experience_buffers[7].append(multi_agent_mask)

            self.episode_step += 1
            self.action_change += info[0]
            self.episode_reward += info[1]
            for k, v in info[5].items():
                self.episode_eval_metrics[k].append(v)

            # s = s1
            observations = next_observations

            if done:
                self.calculate_advantage_values()
                break

        perf_metrics = [self.action_change / self.episode_step]
        perf_metrics += [np.mean(v) for v in list(self.episode_eval_metrics.values())]
        run_time = time.time() - start_time
        print("{} | Reward: {}, Length: {}, Run Time: {:.2f} s".format(self.curr_episode,
                                                                       self.episode_reward,
                                                                       self.episode_step,
                                                                       run_time))

        return [self.episode_reward, self.episode_step] + perf_metrics

    def calculate_advantage_values(self):
        """
        Calculate target values and advantages values
        (1). Input experience buffers contain:
            0. Batch observations (torch.tensor) [b, n, input_dim]
            1. Batch actions (torch.tensor) [b, n]
            2. Batch neighbor rewards (numpy array) [b, n]
            3. Batch log prob old (torch.tensor) [b, n]
            4. Batch neighbor actions (torch.tensor) [b, n, 4]
            5. Batch policy (torch.tensor) [b, n, output_dim]
            6. Batch target queue vector (torch.tensor) [b, n, 24]
            7. Batch neighbor mask (torch.tensor) [b, n, 4]
        (2). Boostrap_values (numpy array) [n]
        (3). Output returned buffers contain:
            0. Batch observations (torch.tensor) [b, n, input_dim]
            1. Batch actions (torch.tensor) [b, n]
            2. Batch log prob old (torch.tensor) [b, n]
            3. Batch neighbor actions (torch.tensor) [b, n, 4]
            4. Batch target queue vector (torch.tensor) [b, n, 24]
            5. Batch neighbor mask (torch.tensor) [b, n, 4]
            6. Batch values (torch.tensor) [b, n, 1]
            7. Batch advantages (numpy array) [b-1, n]
        """
        gamma = NETWORK_PARAMS.GAMMA

        # [b, n, input_dim]
        batch_obs = torch.stack(self.experience_buffers[0])
        # [b, n]
        batch_actions = torch.stack(self.experience_buffers[1])
        # [b, n]
        batch_log_p_old = torch.stack(self.experience_buffers[3])
        # [b, n, 4]
        batch_neighbor_a = torch.stack(self.experience_buffers[4])
        # [b, n, output_dim]
        batch_prob = torch.stack(self.experience_buffers[5])
        # [b, n, 24]
        batch_queue_target = torch.stack(self.experience_buffers[6])
        # [b, n, 4]
        batch_mask = torch.stack(self.experience_buffers[7])

        # [b, n]
        batch_reward = np.array(self.experience_buffers[2])

        with torch.no_grad():
            if self.model_type == 'SOCIALLIGHT':
                # [b, n, output_dim]
                multi_agent_q_value, _, _ = self.local_network.forward_v(batch_obs, batch_neighbor_a, [None, None], batch_mask)
                # [b, n, 1]
                multi_agent_value = torch.sum(torch.mul(multi_agent_q_value.clone(), batch_prob), 2, True)
            else:
                # [b, n, 1]
                multi_agent_value, _, _ = self.local_network.forward_v(batch_obs, batch_neighbor_a, [None, None], batch_mask)

        adv_list, tar_v_list, bootstrap_v_list = [], [], []
        for i, tls in enumerate(self.env.rl_agent_id):
            # [b,]
            value_plus = multi_agent_value.clone().numpy()[:, i, :].reshape(-1)
            rewards = batch_reward[:, i]
            reward_plus = np.append(rewards.reshape(-1)[:-1], value_plus[-1])

            # Calculate target values
            target_values = discount(reward_plus, gamma)[:-1]

            # Calculate advantages
            deltas = reward_plus[:-1] + gamma * value_plus[1:] - value_plus[:-1]
            advantage_values = discount(deltas, gamma * 0.95)

            tar_v_list.append(target_values)
            adv_list.append(advantage_values)
            bootstrap_v_list.append(value_plus[-1])

        self.returned_buffers[0] = batch_obs
        self.returned_buffers[1] = batch_actions
        self.returned_buffers[2] = batch_log_p_old
        self.returned_buffers[3] = batch_neighbor_a
        self.returned_buffers[4] = batch_queue_target
        self.returned_buffers[5] = batch_mask
        self.returned_buffers[6] = np.stack(adv_list).T
        self.returned_buffers[7] = np.stack(tar_v_list).T

        self.bootstrap_values = np.stack(bootstrap_v_list)

    def job(self, episode_number):
        job_results, metrics = None, None
        self.curr_episode = episode_number
        # Set the local weights to the global weights from the master network
        if DEVICE_PARAMS.LOAD_MODEL:
            weights = torch.load(DEVICE_PARAMS.MODEL_PATH + 'model_MATSC/state_dict.pth', map_location=self.device)
        else:
            weights = torch.load(EXPERIMENT_PARAMS.MODEL_PATH + '/state_dict.pth', map_location=self.device)
        self.set_weights(weights=weights)

        if COMPUTE_TYPE == COMPUTE_OPTIONS.SINGLE_THREADED:
            if JOB_TYPE == JOB_OPTIONS.GET_EXPERIENCE:
                metrics = self.run_episode_single_threaded()
                job_results = [self.returned_buffers, self.bootstrap_values]

            else:
                raise NotImplemented

        elif COMPUTE_TYPE == COMPUTE_OPTIONS.MULTI_THREADED:
            raise NotImplemented

        # Get the job results from the learning agents
        # and send them back to the master network
        info = {"id": self.id}

        return job_results, metrics, info
