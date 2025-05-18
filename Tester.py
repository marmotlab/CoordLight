import os
import time
import torch
import numpy as np
import pandas as pd

from collections import OrderedDict
from torch.distributions.categorical import Categorical
from Utils import check_dir, set_env, set_network, convert_to_tensor, save_as_csv


class Tester(object):
    def __init__(self, meta_agent_id, args_dict):
        self.id = meta_agent_id
        self.args_dict = args_dict
        self.device = torch.device('cuda') if self.args_dict['use_gpu'] else torch.device('cpu')

        self.env = set_env(env_type=self.args_dict['env_type'],
                           server_number=meta_agent_id,
                           test=True)
        self.network = set_network(input_size=self.env.obs_space_n,
                                   output_size=self.env.action_space_n,
                                   agent_size=self.env.agent_space_n,
                                   device=self.device)

        self.curr_episode = None
        self.episode_step = None
        self.action_change = None
        self.episode_reward = None
        self.rnn_states_actor = None
        self.rnn_states_critic = None

        self.test_path = self.args_dict['test_path']
        self.model_path = self.test_path + self.args_dict['experiment_name']

        self.initialization()

    def set_weights(self):
        # weights = torch.load(self.model_path + '/model_MATSC/state_dict.pth', map_location=self.device)
        # self.network.load_state_dict(weights)
        checkpoint = torch.load(self.model_path + '/model_MATSC/checkpoint.pkl', map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()

    def initialization(self):
        # Initialize network params
        self.set_weights()
        # Initialize test path dir
        check_dir(self.test_path)

    def reset_params(self):
        # Initialize variables
        self.episode_step = 0
        self.episode_reward = 0
        self.action_change = 0
        self.rnn_states_actor = None
        self.rnn_states_critic = [None, None]

    def run_episode_single_threaded(self, curr_episode, eval_seed):
        # Initialize variables and buffer
        start_time = time.time()
        self.reset_params()
        self.curr_episode = curr_episode
        self.env.curr_episode = curr_episode
        observations = self.env.reset(eval_seed)
        while True:
            action_dict = OrderedDict()

            multi_agent_obs = np.stack(list(observations.values()))
            multi_agent_obs = convert_to_tensor(data=multi_agent_obs.copy(),
                                                data_type=torch.float32,
                                                device=self.device)
            multi_agent_mask_n = self.env.calculate_neighbor_mask()
            multi_agent_mask = np.stack(list(multi_agent_mask_n.values()))
            multi_agent_mask = convert_to_tensor(multi_agent_mask,
                                                 data_type=torch.float32,
                                                 device=self.device)

            with torch.no_grad():
                multi_agent_policy, _, self.rnn_states_actor = self.network.forward(multi_agent_obs,
                                                                                    self.rnn_states_actor,
                                                                                    multi_agent_mask)
            multi_agent_pi_dist = Categorical(multi_agent_policy)
            multi_agent_actions = multi_agent_pi_dist.sample()

            action_list = multi_agent_actions.cpu().numpy().reshape(-1).tolist()
            for i, action in enumerate(action_list):
                action_dict[self.env.rl_agent_id[i]] = action

            next_observations, rewards, done, info = self.env.step(action_dict=action_dict)

            self.episode_step += 1
            self.action_change += info[0]
            self.episode_reward += info[1]

            # s = s1
            observations = next_observations

            if done:
                break

        # generate evaluation data
        avg_duration = self.calculate_cityflow_eval_data()
        run_time = time.time() - start_time
        print("{} | Reward: {}, Length: {}, Run Time: {:.2f} s".format(self.curr_episode,
                                                                       self.episode_reward,
                                                                       self.episode_step,
                                                                       run_time))

        perf_metrics = [self.episode_reward,
                        self.action_change / self.episode_step,
                        avg_duration]

        return perf_metrics

    def calculate_cityflow_eval_data(self):
        """
        Performance metric calculation function for CityFlow environment
        @return:
        """
        dir_to_log_file = self.model_path + "/veh_info_{}".format(self.curr_episode)
        if not os.path.exists(dir_to_log_file):
            os.makedirs(dir_to_log_file)
        for id in self.env.rl_agent_id:
            # get_dic_vehicle_arrive_leave_time
            dic_veh = self.env.tls_dict[id].get_dic_vehicle_arrive_leave_time()

            # save them as csv file
            path_to_log_file = dir_to_log_file + "/vehicle_inter_{0}.csv".format(id)
            df = pd.DataFrame.from_dict(dic_veh, orient='index')
            df.to_csv(path_to_log_file, na_rep="nan")

        # handling csv file using pandas
        df_vehicle_all = []
        for id in self.env.rl_agent_id:
            path_to_log_file = dir_to_log_file + "/vehicle_inter_{0}.csv".format(id)
            # summary items (duration) from csv
            df_vehicle_inter = pd.read_csv(path_to_log_file,
                                           sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                           names=["vehicle_id", "enter_time", "leave_time"])
            df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
            df_vehicle_inter['leave_time'].fillna(3600, inplace=True)
            df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - df_vehicle_inter["enter_time"].values
            ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
            print("------------- id: {0}\tave_duration: {1}\t".format(id, ave_duration))
            df_vehicle_all.append(df_vehicle_inter)
        df_vehicle_all = pd.concat(df_vehicle_all)
        vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
        ave_duration = vehicle_duration.mean()
        print('Episode: {}|| Average travel time : {}'.format(self.curr_episode, ave_duration))

        return ave_duration

    def output_cityflow_eval_data(self, data_list):
        """
        Output and save the evaluation results/average trip time
        """
        with open(self.model_path + "/result.txt", "w+") as f:
            f.write("Average travel time: {0} \n".format(np.nanmean(data_list)))
            f.write("Max travel time: {0} \n".format(np.nanmax(data_list)))
            f.write("Min travel time: {0} \n".format(np.nanmin(data_list)))
            f.write("All average travel time: {0} \n".format(data_list))
        print("======== Generating results =======\n")

    def output_cityflow_step_eval_data(self):
        """
        Output the overall generated traffic and trip evaluation data
        @return:
        """
        traffic_data_path = self.model_path + '/' + 'large_grid_AgentName_traffic.csv'
        save_as_csv(file=traffic_data_path, data=self.env.test_traffic_data)
        print('======= Save Step Evaluation Data ======\n')


if __name__ == '__main__':
    args_dict = dict()

    args_dict['seed'] = 1000
    args_dict['num_test'] = 10
    args_dict['use_gpu'] = True
    args_dict['env_type'] = 'CITYFLOW'
    args_dict['test_path'] = None
    args_dict['experiment_name'] = None

    total_output_list = []
    tester = Tester(meta_agent_id=0, args_dict=args_dict)

    total_outputs = []
    for i in range(args_dict['num_test']):
        seed = args_dict.seed + i * 100
        eval_output = tester.run_episode_single_threaded(curr_episode=i, eval_seed=seed)
        total_outputs.append(eval_output[2])
    tester.output_cityflow_eval_data(total_outputs)
    # tester.output_cityflow_step_eval_data()
