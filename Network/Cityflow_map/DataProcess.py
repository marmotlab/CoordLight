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


map_name_list = ['DJ', 'DH', 'DN']
flow_name_dict = {'DJ': [1, 2, 3], 'DH': [1, 2], 'DN': [1, 2]}

for map_name in map_name_list:
    flow_name_list = flow_name_dict[map_name]
    for flow_name in flow_name_list:
        if map_name == 'DJ':
            road_file = './Jinan/3_4/roadnet_3_4.json'
            if flow_name == 1:
                flow_file = './Jinan/3_4/anon_3_4_jinan_real.json'

            elif flow_name == 2:
                flow_file = './Jinan/3_4/anon_3_4_jinan_real_2000.json'

            elif flow_name == 3:
                flow_file = './Jinan/3_4/anon_3_4_jinan_real_2500.json'
            else:
                raise NotImplementedError

        elif map_name == 'DH':
            road_file = './Hangzhou/4_4/roadnet_4_4.json'
            if flow_name == 1:
                flow_file = './Hangzhou/4_4/anon_4_4_hangzhou_real.json'

            elif flow_name == 2:
                flow_file = './Hangzhou/4_4/anon_4_4_hangzhou_real_5816.json'
            else:
                raise NotImplementedError

        elif map_name == 'DN':
            road_file = './NewYork/28_7/roadnet_28_7.json'
            if flow_name == 1:
                flow_file = './NewYork/28_7/anon_28_7_newyork_real_double.json'

            elif flow_name == 2:
                flow_file = './NewYork/28_7/anon_28_7_newyork_real_triple.json'
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        flow_data = load_json(flow_file)
        road_data = load_json(road_file)['roads']

        lane_data = {}
        for item in road_data:
            points = item['points']
            lane_data[item['id']] = abs(points[0]['x'] - points[1]['x']) + abs(points[0]['y'] - points[1]['y'])

        total_time = 3600
        time_interval = 60
        time_segment = 1200
        time_line = [time_interval * (i + 1) for i in range(total_time // time_interval)]
        time_seg_line = [time_segment * (i + 1) for i in range(total_time // time_segment)]
        flow_num = np.zeros_like(time_line)
        flow_num_seg = np.zeros_like(time_seg_line)
        start_time_list = np.zeros(3600)
        travel_dist_list = []
        travel_num_int_list = []
        for i, data in enumerate(flow_data):
            route = data['route']
            dist = 0
            travel_num_int_list.append(len(route))
            for road in route:
                dist += lane_data[road]
            travel_dist_list.append(dist)
            start_int_id = data['route'][0][5:8]
            end_int_id = data['route'][-1][5:8]
            start_time = data['startTime']
            index = start_time // time_interval
            index_seg = start_time // time_segment
            if start_time >= total_time:
                break
            else:
                flow_num[index] += 1
                flow_num_seg[index_seg] += 1
                start_time_list[start_time] += 1

        print(map_name, flow_name)
        print("Flow rate: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(np.mean(flow_num),
                                                                 np.std(flow_num),
                                                                 np.max(flow_num),
                                                                 np.min(flow_num)))
        print("Travel Distance: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(np.mean(travel_dist_list),
                                                                       np.std(travel_dist_list),
                                                                       np.max(travel_dist_list),
                                                                       np.min(travel_dist_list)))
        print("Travel Num of Roads: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(np.mean(travel_num_int_list),
                                                                           np.std(travel_num_int_list),
                                                                           np.max(travel_num_int_list),
                                                                           np.min(travel_num_int_list)))
        print("Total Num of Vehicles: {}".format(len(travel_num_int_list)))
        print()
