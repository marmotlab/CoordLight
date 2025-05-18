import json
import numpy as np


def load_json(file):
    """
    load saved data from json file
    @ params: file: the file that will be loaded
    """
    with open(file, 'r+') as file:
        content = file.read()

    return json.loads(content)


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


def get_lane_length(road_net_file_path):
    """
    Retrieve lane length dictionary for the given network
    Return: dict{lanes} normalized with the min lane length
    """
    net_data = load_json(road_net_file_path)
    roads = net_data['roads']

    lane_length_dict = {}
    lane_normalize_factor = {}
    for road in roads:
        points = road["points"]
        road_length = abs(points[0]['x'] + points[0]['y'] - points[1]['x'] - points[1]['y'])
        for i in range(3):
            lane_id = road['id'] + "_{0}".format(i)
            lane_length_dict[lane_id] = road_length

    min_length = min(lane_length_dict.values())
    for key, value in lane_length_dict.items():
        lane_normalize_factor[key] = value / min_length

    return lane_normalize_factor, lane_length_dict


map_name_list = ['DJ', 'DH', 'DN']
for map_name in map_name_list:
    if map_name == 'DJ':
        road_file = './Jinan/3_4/roadnet_3_4.json'
        config_file = './Jinan/3_4/roadnet_3_4_net_config.json'

    elif map_name == 'DH':
        road_file = './Hangzhou/4_4/roadnet_4_4.json'
        config_file = './Hangzhou/4_4/roadnet_4_4_net_config.json'

    elif map_name == 'DN':
        road_file = './NewYork/28_7/roadnet_28_7.json'
        config_file = './NewYork/28_7/roadnet_28_7_net_config.json'

    else:
        raise NotImplementedError

# map_name_list = ['3_3', '6_6', '6_6', '10_10', '33_33']
# for map_name in map_name_list:
#     if map_name == '3_3':
#         road_file = './template_lsr/3_3/roadnet_3_3.json'
#         config_file = './template_lsr/3_3/roadnet_3_3_net_config.json'
#
#     elif map_name == '6_6':
#         road_file = './template_lsr/6_6/roadnet_6_6.json'
#         config_file = './template_lsr/6_6/roadnet_6_6_net_config.json'
#
#     elif map_name == '10_10':
#         road_file = './template_lsr/10_10/roadnet_10_10.json'
#         config_file = './template_lsr/10_10/roadnet_10_10_net_config.json'
#
#     elif map_name == '33_33':
#         road_file = './template_lsr/33_33/roadnet_33_33.json'
#         config_file = './template_lsr/33_33/roadnet_33_33_net_config.json'
#
#     else:
#         raise NotImplementedError

    all_lane_normalize_factor, all_lane_length_dict = get_lane_length(road_file)
    config_data = load_json(config_file)
    for tls, data in config_data.items():
        lane_normalize_factor, lane_length_dict = {}, {}
        for in_lane in data['incoming_lane_list']:
            lane_normalize_factor[in_lane] = all_lane_normalize_factor[in_lane]
            lane_length_dict[in_lane] = all_lane_length_dict[in_lane]
        for out_lane in data['outgoing_lane_list']:
            lane_normalize_factor[out_lane] = all_lane_normalize_factor[out_lane]
            lane_length_dict[out_lane] = all_lane_length_dict[out_lane]

        data['lane_length_dict'] = lane_length_dict

    save_as_json(config_file, config_data)
    print("Update config file for {} network!".format(map_name))
