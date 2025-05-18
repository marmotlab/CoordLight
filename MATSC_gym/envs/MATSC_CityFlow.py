import os
import gym
import json
import numpy as np
import cityflow as engine

from abc import ABC
from Utils import load_json
from collections import OrderedDict
from Parameters import CITYFLOW_PARAMS
from MATSC_gym.envs.Tls_Cityflow import Tls


class MATSC_CITYFLOW(gym.Env, ABC):
    """
    Gym environment for CityFlow Simulator
    """

    def __init__(self, server_number, test=False):
        self.server_number = server_number
        self.test = test

        self.eng = None
        self.agent_id = None
        self.rl_agent_id = None

        self.obs_space_n = None
        self.action_space_n = None
        self.agent_space_n = None

        self.tls_dict = {}
        self.pre_actions_dict = {}
        self.lane_length_dict = {}
        self.rl_step = 0
        self.cityflow_step = 0
        self.curr_episode = 0
        self.test_traffic_data = []

        self.env_params = CITYFLOW_PARAMS
        self.obs_type = self.env_params.OBS_TYPE
        self.obs_share = self.env_params.OBS_SHARING
        self.action_space_n = self.env_params.PHASE_SIZE
        self.neighbor_reward_type = self.env_params.NEIGHBOR_R_TYPE
        self.sim_green_duration = self.env_params.GREEN_DURATION
        self.sim_yellow_duration = self.env_params.YELLOW_DURATION

        self.pre_veh_list_lane_dict = OrderedDict()

        self.initialization()

    def initialization(self):
        """
        Initialize the network configuration
        """
        # Set up simulation
        self.set_world()

        # Load net configurations
        net_config_data = load_json(self.env_params.CONFIG_FILE)
        self.agent_id = self.rl_agent_id = list(net_config_data.keys())
        for id in self.rl_agent_id:
            self.tls_dict[id] = Tls(tls_id=id, tls_config_data=net_config_data[id])
            self.lane_length_dict.update(self.tls_dict[id].lane_length_dict)
        # The real length of a lane needs to minus the intersection length (around 15 m)
        for lane, length in self.lane_length_dict.items():
            self.lane_length_dict[lane] = length - 15

        if self.obs_share:
            self.obs_space_n = [5]
        else:
            self.obs_space_n = []

        # Specify the dimension for different observations
        if self.obs_type == 'VC':
            self.obs_space_n += [8 + 12 + 5]

        elif self.obs_type == 'SVC':
            self.obs_space_n += [8 + 12 + 5]

        elif self.obs_type == 'EVC':
            self.obs_space_n += [8 + 12 + 5]

        elif self.obs_type == 'GP':
            self.obs_space_n += [8 + 36 + 5]

        elif self.obs_type == 'EP':
            self.obs_space_n += [8 + 12 + 5]

        elif self.obs_type == 'ATS':
            self.obs_space_n += [8 + 12 + 12 + 5]

        elif self.obs_type == 'DTSE':
            self.obs_space_n += [600 + 5]

        elif self.obs_type == 'QDSE':
            self.obs_space_n += [72 + 5]

        else:
            raise NotImplementedError

        self.agent_space_n = len(self.rl_agent_id)

    def set_world(self):
        """
        Set up the CityFlow simulator
        """
        cityflow_config = {
            "interval": self.env_params.INTERVAL,
            "seed": self.env_params.SEED,
            "laneChange": self.env_params.LANE_CHANGE,
            "dir": self.env_params.DIR,
            "roadnetFile": self.env_params.ROAD_NET_FILE,
            "flowFile": self.env_params.FLOW_FILE,
            "rlTrafficLight": self.env_params.RL_TRAFFIC_LIGHT,
            "saveReplay": self.env_params.SAVE_REPLAY,
            "roadnetLogFile": self.env_params.ROAD_NET_LOG_FILE.format(self.server_number),
            "replayLogFile": self.env_params.REPLY_LOG_FILE.format(self.server_number)
        }

        # Generate cityflow config file for start the simulator
        config_file_name = "cityflow_{}.config".format(self.server_number)
        with open(os.path.join(CITYFLOW_PARAMS.BASE_PATH, config_file_name), "w") as json_file:
            json.dump(cityflow_config, json_file)

        # Start CityFlow
        self.eng = engine.Engine(os.path.join(CITYFLOW_PARAMS.BASE_PATH, config_file_name), thread_num=1)

    def reset(self, seed=0):
        """
        Resets the environment to an initial state and returns an initial observation.
        @ return: initial observations
        """
        print('Worker {} || Reset Environment'.format(self.server_number))

        # Set random seed
        if self.test:
            self.eng.set_random_seed(seed)
        else:
            if self.env_params.RANDOM_SEED:
                seed = np.random.randint(0, 10000)
            else:
                seed = self.env_params.SEED
            self.eng.set_random_seed(seed)

        # Reset CityFlow
        self.eng.reset()
        # self.set_world()
        self.reset_vars()

        return self.observe()

    def reset_vars(self):
        # Reset variables
        for id in self.agent_id:
            self.pre_actions_dict[id] = None
            self.tls_dict[id].reset_vars()
        self.rl_step = 0
        self.cityflow_step = 0

        self.pre_veh_list_lane_dict = OrderedDict()

    def step(self, action_dict):
        """
        Run one RL time step of the environment's dynamics.
        @ params: action_dict
        @ returns
        """
        change_action_list = []
        sum_action_change = 0

        for id in self.agent_id:
            if self.pre_actions_dict[id] is not None and self.pre_actions_dict[id] != action_dict[id]:
                change_action_list.append(id)
                if id in self.rl_agent_id:
                    sum_action_change += 1

        for t in range(self.sim_green_duration):
            for id in self.agent_id:
                if t == 0:
                    if id in change_action_list:
                        self.tls_dict[id].counter['yellow'] = self.sim_yellow_duration
                        self.tls_dict[id].counter['green'] = self.sim_green_duration - self.sim_yellow_duration
                    else:
                        self.tls_dict[id].counter['yellow'] = 0
                        self.tls_dict[id].counter['green'] = self.sim_green_duration

                if self.tls_dict[id].counter['yellow'] > 0:
                    self.tls_dict[id].counter['yellow'] -= 1
                    self.set_yellow_phase(id=id, phase_index=0)
                else:
                    self.tls_dict[id].counter['green'] -= 1
                    self.set_green_phase(id=id, phase_index=action_dict[id])

            self.inner_step()

        self.rl_step += 1
        self.pre_actions_dict = action_dict

        next_obs_dict = self.observe()
        rewards_dict = self.get_rewards()
        neighbor_reward_dict, queue_reward_vector = self.calculate_regional_reward()
        eval_metrics_step_dict = self.measure_traffic_step()
        done = self.check_terminal()
        avg_action_change = sum_action_change / int(len(self.rl_agent_id))
        avg_multi_agent_reward = sum(list(rewards_dict.values())) / int(len(self.rl_agent_id))
        avg_multi_agent_neighbor_reward = sum(list(neighbor_reward_dict.values())) / int(len(self.rl_agent_id))

        return next_obs_dict, rewards_dict, done, [avg_action_change,
                                                   avg_multi_agent_reward,
                                                   avg_multi_agent_neighbor_reward,
                                                   queue_reward_vector,
                                                   neighbor_reward_dict,
                                                   eval_metrics_step_dict]

    def set_yellow_phase(self, id, phase_index):
        """
        Set yellow phase
        """
        self.eng.set_tl_phase(id, phase_index)

    def set_green_phase(self, id, phase_index):
        """
        Set green phase
        """
        self.tls_dict[id].curr_phase = phase_index
        self.eng.set_tl_phase(id, phase_index + 1)

    def inner_step(self):
        """
        Run one CityFLow step
        """
        if self.test:
            global_curr_time = self.eng.get_current_time()
            global_lanes_veh_id_dict = self.eng.get_lane_vehicles()
            global_lanes_num_veh_dict = self.eng.get_lane_vehicle_count()
            global_lanes_num_wait_veh_dict = self.eng.get_lane_waiting_vehicle_count()
            for id in self.agent_id:
                self.tls_dict[id].current_time = global_curr_time
                self.tls_dict[id].global_lanes_veh_id_dict = global_lanes_veh_id_dict
                self.tls_dict[id].global_lanes_num_veh_dict = global_lanes_num_veh_dict
                self.tls_dict[id].global_lanes_num_wait_veh_dict = global_lanes_num_wait_veh_dict
                self.tls_dict[id].get_vehicle_list()
            # self.measure_traffic_step_test()

        self.eng.next_step()
        self.cityflow_step += 1

    def observe(self):
        """
        Calculate observation vectors:
        1. Observation option 1: Vehicle Count (VC), with the shape of (20):
            (1). The current phase of the traffic light
            (2). The number of running vehicles in each incoming lane
        2. Observation option 2: Stopped Vehicle Count (SVC), with the shape of (20):
            (1). The current phase of the traffic light
            (2). The number of halting vehicles in each incoming lane
            (2). The number of running vehicles within the effective range of each incoming lane
        3. Observation option 3: General Pressure (GP), with the shape of (44):
            (1). The current phase of the traffic light
            (2). The pressure of each traffic movement (one incoming -> one outgoing lane)
        4. Observation option 4: Efficient Pressure (EP), with the shape of (20):
            (1). The current phase of the traffic light
            (2). The pressure of each traffic movement (average on incoming lanes and average on outgoing lanes)
        5. Observation option 5: Advanced Traffic State (ATS), with the shape of (32):
            (1). The current phase of the traffic light
            (2). The pressure of each traffic movement within the effective range
            (3). The running vehicle count of each incoming lane within the effective range
        6. Observation option 6: Queueing Dynamic State Encoding (QDSE), with the shape of (72)
        7. Observation option 7: Discrete Traffic State Encoding (DTSE), with the shape of (600)
        @return:
        """
        obs_dict = OrderedDict()
        if self.obs_type == 'VC':
            # Vehicle count, with the shape of [8+12]
            num_veh_lane_dict = self.eng.get_lane_vehicle_count()
            for id in self.agent_id:
                phase_state = np.zeros(8)
                number_veh = np.zeros(self.tls_dict[id].num_in_lane)

                phase_state[self.tls_dict[id].curr_phase] = 1
                for i, in_lane in enumerate(self.tls_dict[id].incoming_lane_list):
                    number_veh[i] = num_veh_lane_dict[in_lane] / 50

                obs_dict[id] = np.concatenate((phase_state, number_veh))

        elif self.obs_type == 'EVC':
            # Efficient counts of vehicles, with shape of [8+12]
            veh_list_lane_dict = self.eng.get_lane_vehicles()
            veh_dist_dict = self.eng.get_vehicle_distance()
            veh_max_speed = 11.11
            efficient_range = veh_max_speed * self.sim_green_duration

            for id in self.agent_id:
                phase_state = np.zeros(8)
                phase_state[self.tls_dict[id].curr_phase] = 1

                efficient_num_running_veh = np.zeros(self.tls_dict[id].num_in_lane)
                for i, in_lane in enumerate(self.tls_dict[id].incoming_lane_list):
                    curr_veh_list = veh_list_lane_dict[in_lane]
                    lane_length = self.lane_length_dict[in_lane]
                    for v in curr_veh_list:
                        if (lane_length - veh_dist_dict[v]) <= efficient_range:
                            efficient_num_running_veh[i] += 1

                efficient_num_running_veh = efficient_num_running_veh / 50
                obs_dict[id] = np.concatenate((phase_state, efficient_num_running_veh))

        elif self.obs_type == 'SVC':
            # Stopped vehicle count, with shape of [8+12]
            num_halting_veh_lane_dict = self.eng.get_lane_waiting_vehicle_count()
            for id in self.agent_id:
                phase_state = np.zeros(8)
                queue_length = np.zeros(self.tls_dict[id].num_in_lane)

                phase_state[self.tls_dict[id].curr_phase] = 1
                for i, in_lane in enumerate(self.tls_dict[id].incoming_lane_list):
                    queue_length[i] = num_halting_veh_lane_dict[in_lane] / 50

                obs_dict[id] = np.concatenate((phase_state, queue_length))

        elif self.obs_type == 'GP':
            # General pressure state definition， with shape of [8+36]
            num_halting_veh_lane_dict = self.eng.get_lane_waiting_vehicle_count()
            for id in self.agent_id:
                phase_state = np.zeros(8)
                phase_state[self.tls_dict[id].curr_phase] = 1

                movements_pressure = np.zeros(self.tls_dict[id].num_in_lane * 3)
                for i, in_lane in enumerate(self.tls_dict[id].incoming_lane_list):
                    lane_link_list = self.tls_dict[id].lane_links[in_lane]

                    for j, out_lane in enumerate(lane_link_list):
                        pressure = num_halting_veh_lane_dict[in_lane] - num_halting_veh_lane_dict[out_lane]
                        movements_pressure[i*3+j] = pressure
                movements_pressure = movements_pressure / 50

                obs_dict[id] = np.concatenate((phase_state, movements_pressure))

        elif self.obs_type == 'EP':
            # Efficient pressure with shape of [8 + 12]
            num_halting_veh_lane_dict = self.eng.get_lane_waiting_vehicle_count()
            for id in self.agent_id:
                phase_state = np.zeros(8)
                phase_state[self.tls_dict[id].curr_phase] = 1

                efficient_pressure = np.zeros(self.tls_dict[id].num_in_lane)
                for i, in_lane in enumerate(self.tls_dict[id].incoming_lane_list):
                    lane_link_list = self.tls_dict[id].lane_links[in_lane]
                    num_veh_in = num_halting_veh_lane_dict[in_lane]
                    num_veh_out = 0
                    for j, out_lane in enumerate(lane_link_list):
                        num_veh_out += num_halting_veh_lane_dict[out_lane]
                    efficient_pressure[i] = num_veh_in - num_veh_out / len(lane_link_list)
                efficient_pressure = efficient_pressure / 50

                obs_dict[id] = np.concatenate((phase_state, efficient_pressure))

        elif self.obs_type == 'ATS':
            # ATS with shape of [8 + 12 + 12]
            num_halting_veh_lane_dict = self.eng.get_lane_waiting_vehicle_count()
            veh_list_lane_dict = self.eng.get_lane_vehicles()
            veh_dist_dict = self.eng.get_vehicle_distance()
            veh_speed_dict = self.eng.get_vehicle_speed()
            veh_max_speed = 11.11
            efficient_range = veh_max_speed * self.sim_green_duration
            for id in self.agent_id:
                phase_state = np.zeros(8)
                phase_state[self.tls_dict[id].curr_phase] = 1

                efficient_pressure = np.zeros(self.tls_dict[id].num_in_lane)
                efficient_num_running_veh = np.zeros(self.tls_dict[id].num_in_lane)
                for i, in_lane in enumerate(self.tls_dict[id].incoming_lane_list):
                    curr_veh_list = veh_list_lane_dict[in_lane]
                    lane_length = self.lane_length_dict[in_lane]
                    for v in curr_veh_list:
                        if (lane_length - veh_dist_dict[v]) <= efficient_range:
                            if veh_speed_dict[v] > 0.1:
                                efficient_num_running_veh[i] += 1

                    lane_link_list = self.tls_dict[id].lane_links[in_lane]
                    num_veh_in = num_halting_veh_lane_dict[in_lane]
                    num_veh_out = 0
                    for j, out_lane in enumerate(lane_link_list):
                        num_veh_out += num_halting_veh_lane_dict[out_lane]
                    efficient_pressure[i] = num_veh_in - num_veh_out / len(lane_link_list)

                efficient_pressure = efficient_pressure / 50
                efficient_num_running_veh = efficient_num_running_veh / 50

                obs_dict[id] = np.concatenate((phase_state, efficient_num_running_veh, efficient_pressure))

        elif self.obs_type == 'QDSE':
            obs_dict = self.observe_QDSE(distance=300)

        elif self.obs_type == 'DTSE':
            obs_dict = self.observe_DTSE(distance=300)

        else:
            raise NotImplementedError

        combined_obs_dict = OrderedDict()
        for id in self.agent_id:
            neighbor_index = np.zeros(5)
            neighbor_index[0] = 1
            combined_obs_dict[id] = [np.concatenate((obs_dict[id], neighbor_index))]
            for i, neighbor in enumerate(self.tls_dict[id].neighbor_list):
                neighbor_index = np.zeros(5)
                neighbor_index[i + 1] = 1
                if neighbor is not None:
                    combined_obs_dict[id].append(np.concatenate((obs_dict[neighbor], neighbor_index)))
                else:
                    combined_obs_dict[id].append(np.concatenate((np.zeros_like(obs_dict[id]), neighbor_index)))

        return combined_obs_dict

    def observe_QDSE(self, distance=300):
        """
        Observation function for the proposed Queueing Dynamics State Encoding (QDSE), which contains:
        (1). The number of halting vehicle in each incoming lane
        (2). The number of entering vehicle in each incoming lane
        (3). The number of leaving vehicle in each incoming lane
        (4). The number of active vehicles in each incoming lane
        (5). The number of vehicles followed the leading active vehicle in each incoming lane
        (6). The distance between the leading active vehicle and the end of queue length in each incoming lane
        @params: distance: the effective distance for the virtual detector
        """
        obs_dict = OrderedDict()

        veh_speed_dict = self.eng.get_vehicle_speed()
        veh_dist_dict = self.eng.get_vehicle_distance()
        veh_list_lane_dict = self.eng.get_lane_vehicles()
        num_veh_lane_dict = self.eng.get_lane_vehicle_count()
        num_halting_veh_lane_dict = self.eng.get_lane_waiting_vehicle_count()

        if distance is not None:
            # Filter the vehicles within the distance
            for lane, curr_veh_list in veh_list_lane_dict.items():
                new_num_veh, new_num_halting_veh, new_veh_list, = 0, 0, []
                for veh in curr_veh_list:
                    veh_pos = self.lane_length_dict[lane] - veh_dist_dict[veh]
                    veh_speed = veh_speed_dict[veh]
                    if veh_pos <= distance:
                        new_num_veh += 1
                        new_veh_list.append(veh)
                        if veh_speed <= 1:
                            new_num_halting_veh += 1

                veh_list_lane_dict[lane] = new_veh_list
                num_veh_lane_dict[lane] = new_num_veh
                num_halting_veh_lane_dict[lane] = new_num_halting_veh

        for id in self.agent_id:
            # Initialize arrays for storing traffic statistics per lane
            num_in_lane = self.tls_dict[id].num_in_lane
            queue_length = np.zeros(num_in_lane)
            moving_num_veh = np.zeros(num_in_lane)
            out_num_veh = np.zeros(num_in_lane)
            in_num_veh = np.zeros(num_in_lane)
            first_veh_dist = np.full(num_in_lane, -1)  # Initialize with -1 (for no vehicles)
            first_num_veh = np.zeros(num_in_lane)

            # Process each incoming lane for the traffic light system
            for i, in_lane in enumerate(self.tls_dict[id].incoming_lane_list):
                queue_length[i] = num_halting_veh_lane_dict[in_lane]
                moving_num_veh[i] = num_veh_lane_dict[in_lane] - queue_length[i]

                curr_veh_list = veh_list_lane_dict[in_lane]
                pre_veh_list = self.pre_veh_list_lane_dict.get(in_lane, [])

                # Calculate the number of vehicles moving out of the intersection
                out_num_veh[i] = len(set(pre_veh_list) - set(curr_veh_list))

                # Calculate the number of vehicles moving into the intersection
                in_num_veh[i] = len(set(curr_veh_list) - set(pre_veh_list))

                # Organize vehicles by distance in descending order
                veh_pos_dict_lane = {v: veh_dist_dict[v] for v in curr_veh_list}
                sorted_veh_list = sorted(veh_pos_dict_lane, key=veh_pos_dict_lane.get, reverse=True)

                if sorted_veh_list:
                    running_veh_list = sorted_veh_list[int(queue_length[i]):]
                    if running_veh_list:
                        # # If there are stopped vehicles, the end of queue is the position of the last stopped vehicle
                        # if queue_length[i] > 0:
                        #     end_queue = veh_pos_dict_lane[sorted_veh_list[int(queue_length[i]) - 1]]
                        # # If there is no stopped vehicle, the end of queue is the stop line
                        # else:
                        #     end_queue = self.lane_length_dict[in_lane]
                        end_queue = self.lane_length_dict[in_lane]
                        first_veh_dist[i] = abs(end_queue - veh_pos_dict_lane[running_veh_list[0]])

                        leading_veh_pos = veh_pos_dict_lane[running_veh_list[0]]
                        # Calculate the number of vehicles following the first running vehicle within a distance of 20
                        for v in running_veh_list:
                            veh_pos = veh_pos_dict_lane[v]
                            # Time headway set to 2s for maximum speed 11m/s
                            if abs(leading_veh_pos - veh_pos) <= 20:
                                first_num_veh[i] += 1
                                leading_veh_pos = veh_pos
                            else:
                                break

                # Update previous vehicle list for the lane
                self.pre_veh_list_lane_dict[in_lane] = curr_veh_list

            obs_dict[id] = np.concatenate((queue_length / 25,
                                           moving_num_veh / 25,
                                           out_num_veh / 10,
                                           in_num_veh / 10,
                                           first_veh_dist / 500,
                                           first_num_veh / 10))

        return obs_dict

    def observe_DTSE(self, distance=300, grid_size=6):
        """
        Observation function for the image-like state representation - Discrete Traffic State Encoding (DTSE),
        which discrete the lane into multiple cells and captures the characteristics of vehicle position and speed.
        with the shape of (distance//grid_size*num_in_lane), by default (600,)
        """
        obs_dict = OrderedDict()

        # veh_speed_dict = self.eng.get_vehicle_speed()
        veh_dist_dict = self.eng.get_vehicle_distance()
        veh_list_lane_dict = self.eng.get_lane_vehicles()
        for id in self.agent_id:
            lane_obs = []
            for i, in_lane in enumerate(self.tls_dict[id].incoming_lane_list):
                lane_length = self.lane_length_dict[in_lane]
                lane_veh_list = veh_list_lane_dict[in_lane]
                num_grids = distance // grid_size
                lane_state = [0] * num_grids
                for veh in lane_veh_list:
                    # veh_speed = veh_speed_dict[veh]
                    veh_travel_dist = veh_dist_dict[veh]
                    veh_pos = lane_length - veh_travel_dist
                    # Check if the vehicle is within the first 300 meters
                    if veh_pos <= distance:
                        # Calculate the index of the grid cell occupied by the vehicle
                        grid_index = int(veh_pos // grid_size)
                        lane_state[grid_index] += 1
                lane_obs.append(lane_state)

            obs_dict[id] = np.array(lane_obs).flatten()

        return obs_dict

    def calculate_regional_reward(self, distance=300):
        truth_queue = OrderedDict()
        regional_reward_dict = OrderedDict()

        veh_speed_dict = self.eng.get_vehicle_speed()
        veh_dist_dict = self.eng.get_vehicle_distance()
        veh_list_lane_dict = self.eng.get_lane_vehicles()
        num_veh_lane_dict = self.eng.get_lane_vehicle_count()
        num_halting_veh_lane_dict = self.eng.get_lane_waiting_vehicle_count()

        if distance is not None:
            # Filter the vehicles within the distance
            for lane, curr_veh_list in veh_list_lane_dict.items():
                new_num_veh, new_num_halting_veh, new_veh_list, = 0, 0, []
                for veh in curr_veh_list:
                    veh_pos = self.lane_length_dict[lane] - veh_dist_dict[veh]
                    veh_speed = veh_speed_dict[veh]
                    if veh_pos <= distance:
                        new_num_veh += 1
                        new_veh_list.append(veh)
                        if veh_speed <= 0.1:
                            new_num_halting_veh += 1

                veh_list_lane_dict[lane] = new_veh_list
                num_veh_lane_dict[lane] = new_num_veh
                num_halting_veh_lane_dict[lane] = new_num_halting_veh

        if self.neighbor_reward_type == 'REGION':
            for id in self.agent_id:
                in_lane_queue = []
                out_lane_queue = []
                truth_queue[id] = []
                for in_lane in self.tls_dict[id].incoming_lane_list:
                    in_lane_queue.append(num_halting_veh_lane_dict[in_lane])

                for out_lane in self.tls_dict[id].outgoing_lane_list:
                    out_lane_queue.append(num_halting_veh_lane_dict[out_lane])

                regional_reward_dict[id] = -1 * (np.sum(in_lane_queue) + np.sum(out_lane_queue)) / 50

        elif self.neighbor_reward_type == 'NEIGHBOR':
            for id in self.agent_id:
                neighbor_in_lane_queue = []
                for in_lane in self.tls_dict[id].incoming_lane_list:
                    neighbor_in_lane_queue.append(num_halting_veh_lane_dict[in_lane])

                for neighbor in self.tls_dict[id].neighbor_list:
                    if neighbor is not None:
                        for in_lane in self.tls_dict[neighbor].incoming_lane_list:
                            neighbor_in_lane_queue.append(num_halting_veh_lane_dict[in_lane])

                regional_reward_dict[id] = -1 * np.sum(neighbor_in_lane_queue) / 100

        else:
            raise NotImplementedError

        for tls_id in self.agent_id:
            truth_queue[tls_id] = []
            for in_lane in self.tls_dict[tls_id].incoming_lane_list:
                truth_queue[tls_id].append(num_halting_veh_lane_dict[in_lane] / 50)

            for out_lane in self.tls_dict[tls_id].outgoing_lane_list:
                truth_queue[tls_id].append(num_halting_veh_lane_dict[out_lane] / 50)

            # [24]
            truth_queue[tls_id] = np.array(truth_queue[tls_id]).flatten()

        return regional_reward_dict, truth_queue

    def calculate_target_queue(self):
        """
        Calculate all target queue values for predictions
        @return: dictionary, key/tls_id, value/target queue values
        """
        truth_queue = OrderedDict()
        num_halting_veh_lane_dict = self.eng.get_lane_waiting_vehicle_count()
        for tls_id in self.agent_id:
            truth_queue[tls_id] = []
            for in_lane in self.tls_dict[tls_id].incoming_lane_list:
                truth_queue[tls_id].append(num_halting_veh_lane_dict[in_lane] / 50)

            for out_lane in self.tls_dict[tls_id].outgoing_lane_list:
                truth_queue[tls_id].append(num_halting_veh_lane_dict[out_lane] / 50)

            # [24]
            truth_queue[tls_id] = np.array(truth_queue[tls_id]).flatten()

        return truth_queue

    def calculate_queue_vectors(self):
        """
        Calculate queue vectors together with neighbor intersections
        @return:
        """
        truth_queue = OrderedDict()
        for tls_id in self.agent_id:
            truth_queue[tls_id] = self.tls_dict[tls_id].calculate_target_queue(norm=True)

        queue_vectors_neighbor = OrderedDict()
        for tls_id in self.agent_id:
            queue_vectors_neighbor[tls_id] = [truth_queue[tls_id]]
            for neighbor in self.tls_dict[tls_id].neighbor_list:
                if neighbor is not None:
                    queue_vectors_neighbor[tls_id].append(truth_queue[neighbor])
                else:
                    queue_vectors_neighbor[tls_id].append(np.zeros_like(truth_queue[tls_id]))
            queue_vectors_neighbor[tls_id] = np.array(queue_vectors_neighbor[tls_id])

        return queue_vectors_neighbor

    def calculate_neighbor_reward(self):
        """
        Calculate the regional reward for each agent:
        1. Neighbor reward V1 (CoordLight): the number of halting vehicles of incoming lanes and outgoing lanes
        2. Neighbor reward v2: the number of haling vehicles on the incoming lanes of all neighborhood agents
        """
        neighbor_reward_dict = OrderedDict()
        num_halting_veh_lane_dict = self.eng.get_lane_waiting_vehicle_count()
        if self.neighbor_reward_type == 'REGION':
            for id in self.agent_id:
                in_lane_queue = []
                out_lane_queue = []
                for in_lane, out_lane in zip(self.tls_dict[id].incoming_lane_list,
                                             self.tls_dict[id].outgoing_lane_list):
                    in_lane_queue.append(num_halting_veh_lane_dict[in_lane])
                    out_lane_queue.append(num_halting_veh_lane_dict[out_lane])

                neighbor_reward_dict[id] = -1 * (np.mean(in_lane_queue) + np.mean(out_lane_queue)) / 5

        elif self.neighbor_reward_type == 'NEIGHBOR':
            for id in self.agent_id:
                neighbor_in_lane_queue = []
                for in_lane in self.tls_dict[id].incoming_lane_list:
                    neighbor_in_lane_queue.append(num_halting_veh_lane_dict[in_lane])

                for neighbor in self.tls_dict[id].neighbor_list:
                    if neighbor is not None:
                        for in_lane in self.tls_dict[neighbor].incoming_lane_list:
                            neighbor_in_lane_queue.append(num_halting_veh_lane_dict[in_lane])

                neighbor_reward_dict[id] = -1 * np.sum(neighbor_in_lane_queue) / 100

        else:
            raise NotImplementedError

        return neighbor_reward_dict

    def calculate_neighbor_mask(self):
        """
        Calculate the attention mask for neighboring intersections
        @return:
        """
        mask_n = OrderedDict()
        for tls in self.agent_id:
            mask_n[tls] = []
            for neighbor in self.tls_dict[tls].neighbor_list:
                if neighbor is not None:
                    mask_n[tls].append(0)
                else:
                    mask_n[tls].append(1)
            mask_n[tls] = np.array(mask_n[tls])

        return mask_n

    def get_rewards(self):
        """
        Calculate individual rewards for each agent
        @ return: dict
        """
        rewards = OrderedDict()
        # Retrieve global variables
        lanes_num_halting_veh_dict = self.eng.get_lane_waiting_vehicle_count()
        for id in self.agent_id:
            self.tls_dict[id].global_lanes_num_wait_veh_dict = lanes_num_halting_veh_dict
            rewards[id] = self.tls_dict[id].get_reward()

        return rewards

    def get_neighbor_actions(self, action_dict, only_neighbor=True):
        """
        Get the vectors of neighbors' actions
        """
        neighbor_action_vec_dict = OrderedDict()
        for id in self.agent_id:
            if only_neighbor:
                output_vec = []
            else:
                output_vec = [action_dict[id]]

            for neighbor in self.tls_dict[id].neighbor_list:
                if neighbor is not None:
                    output_vec.append(action_dict[neighbor] + 1)

                else:
                    output_vec.append(-1 + 1)

            neighbor_action_vec_dict[id] = output_vec

        return neighbor_action_vec_dict

    def check_terminal(self):
        """
        Check terminal condition
        """
        done = False
        if self.test:
            if self.eng.get_current_time() > CITYFLOW_PARAMS.MAX_SIMULATION_STEP - 1:
                done = True
        else:
            if self.eng.get_current_time() > CITYFLOW_PARAMS.MAX_SIMULATION_STEP - 1:
                done = True
            elif self.eng.get_vehicle_count() <= 0:
                done = True
        return done

    def measure_traffic_step(self):
        """
        Traffic metrics measurements for each RL step
        """
        if CITYFLOW_PARAMS.MEASURE_STEP is not None:
            # veh_list = self.eng.get_vehicles(include_waiting=False)
            # num_tot_car = len(veh_list)
            avg_speed = np.mean(list(self.eng.get_vehicle_speed().values()))
            avg_queue = np.mean(list(self.eng.get_lane_waiting_vehicle_count().values()))
            std_queue = np.std(list(self.eng.get_lane_waiting_vehicle_count().values()))
            avg_travel_time = np.mean(np.array(self.eng.get_average_travel_time()))

        else:
            avg_speed = 0
            avg_queue = 0
            std_queue = 0
            avg_travel_time = 0

        curr_traffic = OrderedDict(avg_queue=avg_queue,
                                   std_queue=std_queue,
                                   avg_speed=avg_speed,
                                   avg_travel=avg_travel_time)

        return curr_traffic

    def measure_traffic_step_test(self):
        """
        Measure traffic metrics per sumo step only for evaluation/test
        @return: dictionary, key/metric name, value/metric value
        1. avg_speed_mps: average speed for the whole network
        2. avg_wait_sec: average waiting time for the whole network
        3. avg_queue: average queue for the whole network
        4. std_queue: stand deviation of queue for the whole network
        @warning: activating this function will slow down the simulation
        """
        time_step = self.eng.get_current_time()
        avg_travel_time = self.eng.get_average_travel_time()
        veh_speed_dict = self.eng.get_vehicle_speed()
        lane_queue_dict = self.eng.get_lane_waiting_vehicle_count()

        veh_list = list(veh_speed_dict.keys())
        num_tot_car = len(veh_list)
        if num_tot_car > 0:
            avg_speed = np.mean(list(veh_speed_dict.values()))
        else:
            avg_speed = 0

        avg_queue = np.mean(list(lane_queue_dict.values()))
        std_queue = np.std(list(lane_queue_dict.values()))

        curr_traffic = {'episode': self.curr_episode,
                        'time_sec': time_step,
                        'number_total_car': num_tot_car,
                        'avg_speed_mps': avg_speed,
                        'std_queue': std_queue,
                        'avg_queue': avg_queue,
                        'travel_time': avg_travel_time}

        self.test_traffic_data.append(curr_traffic)


if __name__ == '__main__':

    os.chdir('../..')

    env = MATSC_CITYFLOW(server_number=99)
    obs = env.reset()
    done = False
    ind = 'intersection_2_2'
    while not done:
        actions = {}
        for id in env.agent_id:
            actions[id] = 0
        print("Time: {}".format(env.eng.get_current_time()))
        next_obs, rewards, done, info = env.step(actions)

        lane_wait_veh = env.eng.get_lane_waiting_vehicle_count()
        lane_num_veh = env.eng.get_lane_waiting_vehicle_count()
        veh_dict = env.eng.get_lane_vehicles()

        print(lane_num_veh[env.tls_dict[ind].incoming_lane_list[6]])
        print(lane_wait_veh[env.tls_dict[ind].incoming_lane_list[6]])
        veh_list = veh_dict[env.tls_dict[ind].incoming_lane_list[6]]
