import numpy as np


class Tls():
    """
    Subclass for intersections
    """

    def __init__(self, tls_id, tls_config_data):
        self.id = tls_id
        self.config_data = tls_config_data

        self.incoming_lane_list = self.config_data['incoming_lane_list']
        self.outgoing_lane_list = self.config_data['outgoing_lane_list']
        self.lane_links = self.config_data['lane_links']
        self.action_space = self.config_data['action_space']
        self.neighbor_list = self.config_data['neighbor_list']
        self.lane_length_dict = self.config_data['lane_length_dict']

        self.curr_phase = None
        self.phase_state_one_hot = None
        self.num_veh_in_lane = None
        self.num_veh_out_lane = None
        self.waiting_time_front_veh = None

        # For calculation & Vehicle information
        self.current_time = None
        self.global_lanes_num_veh_dict = None
        self.global_lanes_veh_id_dict = None
        self.global_lanes_num_wait_veh_dict = None
        self.dic_vehicle_arrive_leave_time = dict()
        self.list_lane_vehicle_current_step = []
        self.list_lane_vehicle_previous_step = []

        # Intersection attributes
        self.action_space_n = len(self.action_space) - 1
        self.num_in_lane = len(self.incoming_lane_list)
        self.num_out_lane = len(self.outgoing_lane_list)

        # Phase setting
        self.counter = {'yellow': 0, 'green': 0}

    def reset_vars(self):
        """
        Reset all variables that calculate vehicle information
        """
        self.current_time = None
        self.global_lanes_num_veh_dict = None
        self.global_lanes_veh_id_dict = None
        self.global_lanes_num_wait_veh_dict = None
        self.dic_vehicle_arrive_leave_time = dict()
        self.list_lane_vehicle_current_step = []
        self.list_lane_vehicle_previous_step = []

    def get_reward(self):
        """
        Calculate the rewards based on queue length
        """
        reward = []
        for i, in_lane in enumerate(self.incoming_lane_list):
            reward.append(self.global_lanes_num_wait_veh_dict[in_lane] / 100)
        reward = -1 * np.clip(np.mean(np.array(reward)), 0, 1)

        return reward

    def calculate_target_queue(self, norm=True):
        """
        Calculate the target queue length for each incoming lane
        """
        target_queue = np.zeros(len(self.incoming_lane_list))
        for i, in_lane in enumerate(self.incoming_lane_list):
            target_queue[i] = np.clip(self.global_lanes_num_wait_veh_dict[in_lane] / 100, 0, 1) \
                if norm else self.global_lanes_num_wait_veh_dict[in_lane]

        return target_queue

    def get_dic_vehicle_arrive_leave_time(self):
        """
        Get the dictionary of vehicle arrive and leave time (for test)
        """
        return self.dic_vehicle_arrive_leave_time

    def _update_arrive_time(self, list_vehicle_arrive):
        """
        Update the time for vehicle to enter entering lane (for test)
        """
        ts = self.current_time
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                # print("vehicle: %s already exists in entering lane!"%vehicle)
                # sys.exit(-1)
                pass

    def _update_left_time(self, list_vehicle_left):
        """
        Update the time for vehicle to leave entering lane (for test)
        """
        ts = self.current_time
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")

    def get_vehicle_list(self):
        """
        Get the list of vehicles in the intersection (for test)
        """
        # get vehicle list
        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_previous_step += self.list_lane_vehicle_current_step  # store current step
        self.list_lane_vehicle_current_step = []
        for lane in self.incoming_lane_list:  # renew current step
            self.list_lane_vehicle_current_step += self.global_lanes_veh_id_dict[lane]

        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(
            set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
