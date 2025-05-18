import os
import sys
import math
import json
import torch
import optparse
import numpy as np
import pandas as pd
import scipy.signal as signal

from Parameters import *


def check_SUMO_HOME():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def make_gif(env, f_name):
    command = 'ffmpeg -framerate 5 -i "{tempGifFolder}/step%03d.png" {outputFile}'.format(tempGifFolder=env.gif_folder,
                                                                                          outputFile=f_name)

    os.system(command)

    deleteTempImages = "rm {tempGifFolder}/*".format(tempGifFolder=env.gif_folder)
    os.system(deleteTempImages)
    print("wrote gif")


def convert_to_item(tensor):
    return tensor.cpu().detach().numpy().item()


def convert_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def convert_to_tensor(data, data_type, device):
    if device is None:
        return torch.as_tensor(data=data,
                               dtype=data_type)
    else:
        return torch.as_tensor(data=data,
                               dtype=data_type,
                               device=device)


def normalize_value(data, factor):
    return data / factor


def clip_value(data, min, max):
    return np.clip(data, min, max)


def calculate_list_mean(list, dim):
    if dim is None:
        return np.nanmean(np.array(list))
    else:
        return np.nanmean(np.array(list), dim)


def calculate_list_sum(list, dim):
    if dim is None:
        return np.nansum(np.array(list))
    else:
        return np.nansum(np.array(list), dim)


def set_network(input_size, output_size, agent_size, device, network_type=NETWORK_PARAMS.NET_TYPE,
                actor_lr=NETWORK_PARAMS.A_LR_Q, critic_lr=NETWORK_PARAMS.C_LR_Q):
    """
    Set desired network model
    @param input_size: input dimension/list
    @param output_size: output dimension
    @param agent_size: agent dimension
    @param device: cpu/gpu
    @param network_type: network type
    @param actor_lr: Learning rate for the actor network
    @param critic_lr: Learning rate for the critic network
    @return: network model/torch object
    """
    if network_type == 'COORDLIGHT':
        from Models.CoordLightModel import CoordLight
        network = CoordLight(input_dim=input_size,
                             output_dim=output_size,
                             agent_dim=agent_size,
                             actor_lr=actor_lr,
                             critic_lr=critic_lr).to(device)

    elif network_type == 'SOCIALLIGHT':
        from Models.SocialModel import SocialLightModel
        network = SocialLightModel(input_dim=input_size,
                                   output_dim=output_size,
                                   agent_dim=agent_size,
                                   actor_lr=actor_lr,
                                   critic_lr=critic_lr).to(device)
    else:
        assert 1 == 0, 'Invalid Model Type!'

    return network


def set_env(env_type, server_number, test=False):
    """
    Set gym MATSC environment
    @param env_type: string - environment type
    @param server_number
    @return:
    """
    if env_type == 'CITYFLOW':
        from MATSC_gym.envs.MATSC_CityFlow import MATSC_CITYFLOW
        return MATSC_CITYFLOW(server_number=server_number, test=test)

    else:
        raise NotImplementedError


def create_config_dict():
    """
    Create config dict to save all params
    @return: dictionary
    """
    params_list = [INPUT_PARAMS, EXPERIMENT_PARAMS, DEVICE_PARAMS, NETWORK_PARAMS]
    if INPUT_PARAMS.ENV_TYPE == 'CITYFLOW':
        params_list += [CITYFLOW_PARAMS]

    config_dict = {}
    for params in params_list:
        params_dict = vars(params)
        for k, v in params_dict.items():
            if not k.startswith('__'):
                config_dict[k] = v

    return config_dict


def create_wandb_config_dict():
    params_list = [DEVICE_PARAMS, NETWORK_PARAMS]
    if INPUT_PARAMS.ENV_TYPE == 'CITYFLOW':
        params_list.append(CITYFLOW_PARAMS)

    else:
        raise NotImplementedError

    config_dict = {}
    for params in params_list:
        params_dict = vars(params)
        for k, v in params_dict.items():
            if not k.startswith('__'):
                config_dict[k] = v

    if INPUT_PARAMS.ENV_TYPE == 'SUMO':
        del config_dict['SUMO_CONFIG_PARAMS']

    return config_dict


def create_config_json(path, params):
    """
    create a txt file to record the config dictionary
    @param path: string
    @param params: dictionary, key/param name, value/param value
    """
    file = open(path, 'w')
    for k, v in params.items():
        file.write('{}: {}\n'.format(k, v))
    file.close()


def create_config_logger(path, params):
    """
    create a json file to record the config dictionary
    @param path: string
    @param params: dictionary, key/param name, value/param value
    """
    return save_as_json(file=path, data=params)


def convert_to_combined_vector(input_dict, neighbor_dict):
    """
    Input single agent vector and return the output vectors that combined with neighbors' vectors
    @param input_dict: dictionary, key/tls id, value/input data
    @param neighbor_dict: dictionary, key/tls id, value/neighbor list
    @return: output_dict: dictionary, key/tls id, value/output data
    """
    output_dict = {}
    for id in list(input_dict.keys()):
        combined_output_list = [input_dict[id]]
        neighbor_list = neighbor_dict[id]
        for neighbor_id in neighbor_list:
            if neighbor_id is None:
                # Padding if neighbor is None
                combined_output_list.append(np.zeros(input_dict[id].copy().shape))
            else:
                combined_output_list.append(input_dict[neighbor_id])
        output_dict[id] = combined_output_list

    return output_dict


def set_up_torch_mp(num_process, func, args):
    """
    set up multi processing based on torch
    @ params num_process: the number of process that will start
    @ params func: a function to run
    @ params args: required input parameters
    @ return:
    """
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    processes = []
    for rank in range(num_process):
        p = mp.Process(target=func, args=args)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def save_as_json(file, data):
    """
    Save data to the json file
    @ params: file: the json file that will be saved to
    @ params: data: data that need to be saved
    """
    data = json.dumps(data)
    with open(file, 'w+') as file:
        file.write(data)
    file.close()


def save_as_csv(file, data):
    """
    save the input data to a csv file
    @param file: file path
    @param data: input data
    @return:
    """
    data = pd.DataFrame(data)
    data.to_csv(file)


def load_json(file):
    """
    load saved data from json file
    @ params: file: the file that will be loaded
    """
    with open(file, 'r+') as file:
        content = file.read()

    return json.loads(content)


def calculate_array_attr(array, len):
    """
    function of calculating the std of defined length vectors of a given array
    :param array: target array to be computed, [batch, 6]
    :param len: single vector
    :return: dict key/std, mean, max, min, value/array [batch, 6]
    """
    attr = {'std': [], 'mean': [], 'max': [], 'min': []}
    array = np.array(array)
    batch, lane_dim = array.shape
    for i in range(lane_dim):
        array_std, array_mean, array_max, array_min = [], [], [], []
        array_history_single = array[:, i]
        for j in range(batch):
            if j < batch - len:
                array_std.append(np.std(array_history_single[j:j + len]))
                array_mean.append(np.mean(array_history_single[j:j + len]))
                array_max.append(np.max(array_history_single[j:j + len]))
                array_min.append(np.min(array_history_single[j:j + len]))
            else:
                array_std.append(np.std(array_history_single[j:]))
                array_mean.append(np.mean(array_history_single[j:]))
                array_max.append(np.max(array_history_single[j:]))
                array_min.append(np.min(array_history_single[j:]))
        attr['std'].append(array_std)
        attr['mean'].append(array_mean)
        attr['max'].append(array_max)
        attr['min'].append(array_min)
    for k, v in attr.items():
        attr[k] = np.array(v).T
    return attr


def check_dir(dir):
    """
    Check if the dir exists, otherwise create it
    @param dir:
    @return:
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_dirs(params):
    """
    Create all experiment directories
    @param params: class
    @return:
    """
    for k, v in params.__dict__.items():
        if 'PATH' in k.split('_') and len(k.split('_')) == 2:
            check_dir(dir=v)


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """
    Decreases the learning rate linearly
    """
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def discount_reward(rewards, gamma, final_r, require_grad):
    """
    Calculate discounted reward
    @param rewards: numpy array/tensor -> reward vector
    @param gamma: numpy array/tensor -> discount factor
    @param final_r: numpy array/tensor -> final value for bootstrapping
    @param require_grad:bool -> Check if calculating gradients
    @return:
    """
    if not require_grad:
        discounted_r = np.zeros_like(rewards)
        running_add = final_r
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_r[t] = running_add
    else:
        discounted_r = torch.zeros_like(rewards.clone().detach())
        running_add = torch.reshape(final_r, (1, 1))
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[t].clone().detach()
            discounted_r[t] = running_add

    return discounted_r


def calculate_lambda_return(rewards, values, gamma, lamb):
    """
    Calculate the lambda return via array
    :params rewards: Rewards: numpy array, shape: (seq_len)
    :params values: Values: numpy array, shape: (seq_len)
    :params gamma: Discount factor
    :params lamb: Lambda factor
    :return: Lambda returns (seq_len)
    """
    lamb_return = np.zeros_like(rewards)
    for j in range(rewards.shape[-1]):
        index = rewards.shape[-1] - j - 1

        if j == 0:
            lamb_return[index] = rewards[index] + gamma * values[index + 1]

        else:
            lamb_return[index] = rewards[index] + gamma * (1 - lamb) * values[index + 1] \
                                 + lamb * gamma * lamb_return[index + 1]

    return lamb_return


def calculate_advantages(rewards, values, gamma, mu):
    """
    Calculate the advantages by using General Advantage Estimation (GAE)
    """
    last_advantages = 0
    last_value = values[-1]

    advantages = np.zeros(len(rewards), dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * last_value - values[t]
        last_advantages = delta + gamma * mu * last_advantages
        advantages[t] = last_advantages
        last_value = values[t]

    return advantages


def linear_decay_entropy_factor(initial_entropy_factor, target_entropy_factor, total_steps, max_steps=2000):
    """
    Dynamically adjusts the entropy factor using a linear decay function.

    Args:
    - initial_entropy_factor: Initial entropy factor
    - target_entropy_factor: Target entropy factor
    - total_steps: Total number of steps

    Returns:
    - decayed_entropy_factor: Entropy factor calculated for the current step
    """
    current_step = min(total_steps, max_steps)  # Limit decay to the first 1000 steps
    decayed_entropy_factor = initial_entropy_factor - (initial_entropy_factor - target_entropy_factor) * (current_step / max_steps)
    return max(decayed_entropy_factor, target_entropy_factor)  # Ensure it does not go below the target value
