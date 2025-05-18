import datetime

data = datetime.datetime.now()


class INPUT_PARAMS:
    MAX_EPISODES        = 20000
    NUM_META_AGENTS     = 4  # [4, 8]
    PL_FACTOR           = 1
    VL_FACTOR           = 0.5
    EL_FACTOR           = 0.3
    QL_FACTOR           = 0.003
    NET_TYPE            = 'COORDLIGHT'
    ENV_TYPE            = 'CITYFLOW'
    UPDATE_TYPE         = 'PPO_MAC'
    CITYFLOW_OBS_TYPE   = 'QDSE'  # ['QDSE', 'VC', 'SVC', 'GP', 'EP', 'ATS', 'DTSE']
    CITYFLOW_NEI_R_TYPE = 'NEIGHBOR'  # ['NEIGHBOR', 'REGION']
    CITYFLOW_MAP_TYPE   = 'DJ'
    CITYFLOW_FLOW_TYPE  = '1'  # ['DJ':[1, 2, 3], 'DH':[1, 3] 'DN':[1,2]]
    EVAL_TRAFFIC        = False
    OBS_SHARING         = True
    REWARD_SHARING      = False
    LOAD_MODEL          = False
    MODEL_PATH          = None
    LR_DECAY            = False
    RESET_OPTIM         = False
    SAMPLE_BUFFER       = False
    BUFFER_SIZE         = None


class EXPERIMENT_PARAMS:
    PROJECT_NAME        = 'MATSC_PyTorch'
    BASE_PATH           = 'Train_CityFlow_CoordLight'
    EXPERIMENT_NAME     = '{}_{}_{}_{}_MATSC_PyTorch_CoordLight_DJ1'.format(data.year, data.month, data.day, data.hour)
    EXPERIMENT_PATH     = './{}/{}'.format(BASE_PATH, EXPERIMENT_NAME)
    MODEL_PATH          = EXPERIMENT_PATH + '/model_MATSC'
    GIFS_PATH           = EXPERIMENT_PATH + '/gifs_MATSC'
    TRAIN_PATH          = EXPERIMENT_PATH + '/train_MATSC'
    TRIP_PATH           = EXPERIMENT_PATH + '/trip_info'
    TEMP_GIF_FOLDER     = EXPERIMENT_PATH + '/TEMP_DATA_DO_NOT_TOUCH/'
    CONFIG_FILE_PATH    = EXPERIMENT_PATH + '/Config_MATSC.json'


class DEVICE_PARAMS:
    MAX_EPISODES        = INPUT_PARAMS.MAX_EPISODES
    NUM_META_AGENTS     = INPUT_PARAMS.NUM_META_AGENTS
    LOAD_MODEL          = INPUT_PARAMS.LOAD_MODEL
    MODEL_PATH          = INPUT_PARAMS.MODEL_PATH
    SAMPLE_BUFFER       = INPUT_PARAMS.SAMPLE_BUFFER
    BUFFER_SIZE         = INPUT_PARAMS.BUFFER_SIZE
    WANDB               = False
    OUTPUT_GIFS         = False
    USE_GPU             = True
    NUM_GPU             = 1
    SAVE_MODEL_STEP     = 100
    SUMMARY_WINDOW      = 30


class CITYFLOW_PARAMS:
    EVAL_NET            = INPUT_PARAMS.CITYFLOW_MAP_TYPE
    EVAL_FLOW           = INPUT_PARAMS.CITYFLOW_FLOW_TYPE
    INTERVAL            = 1
    SEED                = 0
    RANDOM_SEED         = True
    LANE_CHANGE         = False
    RL_TRAFFIC_LIGHT    = True
    SAVE_REPLAY         = False
    MAX_SIMULATION_STEP = 3600
    GREEN_DURATION      = 5
    YELLOW_DURATION     = 2
    PHASE_SIZE          = 8
    MEASURE_STEP        = INPUT_PARAMS.EVAL_TRAFFIC
    OBS_TYPE            = INPUT_PARAMS.CITYFLOW_OBS_TYPE
    OBS_SHARING         = INPUT_PARAMS.OBS_SHARING
    REWARD_SHARING      = INPUT_PARAMS.REWARD_SHARING
    NEIGHBOR_R_TYPE     = INPUT_PARAMS.CITYFLOW_NEI_R_TYPE

    if EVAL_NET == 'G6U':
        BASE_FILE   = 'template_lsr/6_6/'
        NET_FILE    = BASE_FILE + 'roadnet_6_6.json'
        FLOW_FILE   = BASE_FILE + 'anon_6_6_300_0.3_uni.json'
        CONFIG_FILE = BASE_FILE + 'roadnet_6_6_net_config.json'

    elif EVAL_NET == 'G6B':
        BASE_FILE   = 'template_lsr/6_6/'
        NET_FILE    = BASE_FILE + 'roadnet_6_6.json'
        FLOW_FILE   = BASE_FILE + 'anon_6_6_300_0.3_bi.json'
        CONFIG_FILE = BASE_FILE + 'roadnet_6_6_net_config.json'

    elif EVAL_NET == 'DN':
        BASE_FILE   = 'NewYork/28_7/'
        NET_FILE    = BASE_FILE + 'roadnet_28_7.json'
        if EVAL_FLOW == '1':
            FLOW_FILE   = BASE_FILE + 'anon_28_7_newyork_real_double.json'
        elif EVAL_FLOW == '2':
            FLOW_FILE   = BASE_FILE + 'anon_28_7_newyork_real_triple.json'
        else:
            raise NotImplementedError
        CONFIG_FILE = BASE_FILE + 'roadnet_28_7_net_config.json'

    elif EVAL_NET == 'DN2':
        BASE_FILE   = 'NewYork/16_3/'
        NET_FILE    = BASE_FILE + 'roadnet_16_3.json'
        FLOW_FILE   = BASE_FILE + 'anon_16_3_newyork_real.json'
        CONFIG_FILE = BASE_FILE + 'roadnet_16_3_net_config.json'

    elif EVAL_NET == 'DH':
        BASE_FILE   = 'Hangzhou/4_4/'
        NET_FILE    = BASE_FILE + 'roadnet_4_4.json'
        if EVAL_FLOW == '1':
            FLOW_FILE   = BASE_FILE + 'anon_4_4_hangzhou_real.json'
        elif EVAL_FLOW == '2':
            FLOW_FILE   = BASE_FILE + 'anon_4_4_hangzhou_real_5734.json'
        elif EVAL_FLOW == '3':
            FLOW_FILE   = BASE_FILE + 'anon_4_4_hangzhou_real_5816.json'
        else:
            raise NotImplementedError
        CONFIG_FILE = BASE_FILE + 'roadnet_4_4_net_config.json'

    elif EVAL_NET == 'DJ':
        BASE_FILE   = 'Jinan/3_4/'
        NET_FILE    = BASE_FILE + 'roadnet_3_4.json'
        if EVAL_FLOW == '1':
            FLOW_FILE   = BASE_FILE + 'anon_3_4_jinan_real.json'
        elif EVAL_FLOW == '2':
            FLOW_FILE   = BASE_FILE + 'anon_3_4_jinan_real_2000.json'
        elif EVAL_FLOW == '3':
            FLOW_FILE   = BASE_FILE + 'anon_3_4_jinan_real_2500.json'
        else:
            raise NotImplementedError
        CONFIG_FILE = BASE_FILE + 'roadnet_3_4_net_config.json'

    elif EVAL_NET == 'G33':
        BASE_FILE   = 'template_lsr/33_33/'
        NET_FILE    = BASE_FILE + 'roadnet_33_33.json'
        if EVAL_FLOW == '1':
            FLOW_FILE   = BASE_FILE + 'anon_33_33_300_0.3_bi.json'
        elif EVAL_FLOW == '2':
            FLOW_FILE   = BASE_FILE + 'anon_33_33_300_0.3_uni.json'
        else:
            raise NotImplementedError
        CONFIG_FILE = BASE_FILE + 'roadnet_33_33_net_config.json'

    else:
        NET_FILE    = None
        FLOW_FILE   = None
        CONFIG_FILE = None

    DIR                 = './'
    BASE_PATH           = './Network/Cityflow_map'
    ROAD_NET_LOG_FILE   = BASE_PATH + "/frontend/web/roadnetLogFile_{}.json"
    REPLY_LOG_FILE      = BASE_PATH + "/frontend/web/replayLogFile_{}.txt"
    ROAD_NET_FILE       = BASE_PATH + '/' + NET_FILE
    FLOW_FILE           = BASE_PATH + '/' + FLOW_FILE
    CONFIG_FILE         = BASE_PATH + '/' + CONFIG_FILE


class NETWORK_PARAMS:
    NET_TYPE            = INPUT_PARAMS.NET_TYPE
    GAMMA               = .98
    LAMBDA              = .60
    A_LR_Q              = 1.e-4
    C_LR_Q              = 3.e-4
    K_EPOCH             = 5  # [5-15]
    EPS_CLIP            = 0.2
    GRAD_CLIP           = 15
    RESET_OPTIM_STEP    = 10000
    PL_FACTOR           = INPUT_PARAMS.PL_FACTOR
    VL_FACTOR           = INPUT_PARAMS.VL_FACTOR
    EL_FACTOR           = INPUT_PARAMS.EL_FACTOR
    QL_FACTOR           = INPUT_PARAMS.QL_FACTOR
    LR_DECAY            = INPUT_PARAMS.LR_DECAY
    RESET_OPTIM         = INPUT_PARAMS.RESET_OPTIM
    UPDATE_TYPE         = INPUT_PARAMS.UPDATE_TYPE


class JOB_OPTIONS:
    GET_EXPERIENCE       = 1
    GET_GRADIENT         = 2


class COMPUTE_OPTIONS:
    MULTI_THREADED       = 1
    SINGLE_THREADED      = 2


class ALGORITHM_OPTIONS:
    A3C                  = 1
    PPO                  = 2


JOB_TYPE                 = JOB_OPTIONS.GET_EXPERIENCE
COMPUTE_TYPE             = COMPUTE_OPTIONS.SINGLE_THREADED
ALGORITHM_TYPE           = ALGORITHM_OPTIONS.PPO
