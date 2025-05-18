import ray
import random

from Utils import *
from Runner import Runner
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

ray.init(num_gpus=DEVICE_PARAMS.NUM_GPU)
print("Hello World !\n")


def write_to_Tensorboard(global_summary, tensorboard_data, curr_episode, plot_means=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric
    if plot_means:
        tensorboard_data = np.array(tensorboard_data)
        tensorboard_data = list(np.mean(tensorboard_data, axis=0))
        policy_loss, value_loss, entropy_loss, a_grad_norm, c_grad_norm, clip_frac, a_pred_loss, c_pred_loss, \
            episode_reward, episode_length, action_change, _, _, _, _ = tensorboard_data

    else:
        first_episode = tensorboard_data[0]
        policy_loss, value_loss, entropy_loss, a_grad_norm, c_grad_norm, clip_frac, a_pred_loss, c_pred_loss, \
            episode_reward, episode_length, action_change, _, _, _, _ = first_episode

    global_summary.add_scalar(tag='Losses/Value Loss', scalar_value=value_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Entropy Loss', scalar_value=entropy_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Actor Grad Norm', scalar_value=a_grad_norm, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Critic Grad Norm', scalar_value=c_grad_norm, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Clip Fraction', scalar_value=clip_frac, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Actor Prediction Loss', scalar_value=a_pred_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Critic Prediction Loss', scalar_value=c_pred_loss, global_step=curr_episode)

    global_summary.add_scalar(tag='Perf/Reward', scalar_value=episode_reward, global_step=curr_episode)
    global_summary.add_scalar(tag='Perf/Episode Length', scalar_value=episode_length, global_step=curr_episode)
    global_summary.add_scalar(tag='Perf/Action change', scalar_value=action_change, global_step=curr_episode)


def write_to_Tensorboard_eval(global_summary, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric
    first_episode = tensorboard_data[0]
    policy_loss, value_loss, entropy_loss, a_grad_norm, c_grad_norm, clip_frac, a_pred_loss, c_pred_loss, \
        episode_reward, episode_length, action_change, \
        avg_queue, std_queue, avg_speed, avg_travel = first_episode

    global_summary.add_scalar(tag='Losses/Value Loss', scalar_value=value_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Entropy Loss', scalar_value=entropy_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Actor Grad Norm', scalar_value=a_grad_norm, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Critic Grad Norm', scalar_value=c_grad_norm, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Clip Fraction', scalar_value=clip_frac, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Actor Prediction Loss', scalar_value=a_pred_loss, global_step=curr_episode)
    global_summary.add_scalar(tag='Losses/Critic Prediction Loss', scalar_value=c_pred_loss, global_step=curr_episode)

    global_summary.add_scalar(tag='Perf/Reward', scalar_value=episode_reward, global_step=curr_episode)
    global_summary.add_scalar(tag='Perf/Episode Length', scalar_value=episode_length, global_step=curr_episode)
    global_summary.add_scalar(tag='Perf/Action change', scalar_value=action_change, global_step=curr_episode)

    global_summary.add_scalar(tag='Eval/Avg Queue', scalar_value=avg_queue, global_step=curr_episode)
    global_summary.add_scalar(tag='Eval/Std Queue', scalar_value=std_queue, global_step=curr_episode)
    global_summary.add_scalar(tag='Eval/Avg Speed', scalar_value=avg_speed, global_step=curr_episode)
    global_summary.add_scalar(tag='Eval/Avg Travel', scalar_value=avg_travel, global_step=curr_episode)


def get_global_train_buffer(all_jobs, buffer_size=8):
    global_buffer = []
    global_metrics = []
    for i in range(buffer_size):
        global_buffer.append([])
    random.shuffle(all_jobs)
    for job in all_jobs:
        job_results, metrics, info = job
        for i in range(len(global_buffer)):
            global_buffer[i].append(job_results[0][i])
        global_metrics.append(metrics)

    # cat all jobs results, agent dim from n to n * meta_agent
    global_buffer[0] = torch.cat(global_buffer[0], 1)
    global_buffer[1] = torch.cat(global_buffer[1], 1)
    global_buffer[2] = torch.cat(global_buffer[2], 1)
    global_buffer[3] = torch.cat(global_buffer[3], 1)
    global_buffer[4] = torch.cat(global_buffer[4], 1)
    global_buffer[5] = torch.cat(global_buffer[5], 1)

    global_buffer[6] = np.concatenate(global_buffer[6], 1)
    global_buffer[7] = np.concatenate(global_buffer[7], 1)

    if DEVICE_PARAMS.SAMPLE_BUFFER:
        # random sample buffers based on the given ratio
        num_agents = global_buffer[0].shape[1]
        sample_num_agents = DEVICE_PARAMS.BUFFER_SIZE
        assert sample_num_agents is not None
        sample_indices = random.sample(range(num_agents), sample_num_agents)
        # sample the buffers based on the indices
        global_buffer[0] = global_buffer[0][:, sample_indices]
        global_buffer[1] = global_buffer[1][:, sample_indices]
        global_buffer[2] = global_buffer[2][:, sample_indices]
        global_buffer[3] = global_buffer[3][:, sample_indices]
        global_buffer[4] = global_buffer[4][:, sample_indices]
        global_buffer[5] = global_buffer[5][:, sample_indices]

        global_buffer[6] = global_buffer[6][:, sample_indices]
        global_buffer[7] = global_buffer[7][:, sample_indices]

        global_metrics = np.mean(global_metrics, 0)
    else:
        global_metrics = np.mean(global_metrics, 0)

    return global_buffer, global_metrics


def calculate_gradients_ma_ppo(network, device, experience_buffers, norm_adv=True):
    k_epoch = NETWORK_PARAMS.K_EPOCH
    eps_clip = NETWORK_PARAMS.EPS_CLIP
    grad_clip = NETWORK_PARAMS.GRAD_CLIP
    vl_factor = NETWORK_PARAMS.VL_FACTOR
    el_factor = NETWORK_PARAMS.EL_FACTOR
    ql_factor = NETWORK_PARAMS.QL_FACTOR
    model_type = NETWORK_PARAMS.NET_TYPE
    num_meta = DEVICE_PARAMS.NUM_META_AGENTS

    v_l, p_l, e_l, a_gn, c_gn, c_f, a_p_l, c_p_l = 0, 0, 0, 0, 0, 0, 0, 0

    # [b, n, input_dim]
    batch_obs = experience_buffers[0].to(device)
    # [b, n]
    batch_actions = experience_buffers[1].to(device)
    # [b, n]
    batch_log_p_old = experience_buffers[2].to(device)
    # [b, n, 4]
    batch_neighbor_a = experience_buffers[3].to(device)
    # [b, n, 24]
    batch_target_queue = experience_buffers[4].to(device)
    # [b, n, 4]
    batch_mask = experience_buffers[5].to(device)

    # [b-1, n]
    if norm_adv:
        batch_advantages = experience_buffers[6]
        adv_mean = np.nanmean(batch_advantages)
        adv_std = np.nanstd(batch_advantages)
        batch_advantages = (batch_advantages - adv_mean) / (adv_std + 1e-6)
    else:
        batch_advantages = experience_buffers[6]

    batch_advantages = convert_to_tensor(data=batch_advantages.copy(),
                                         data_type=torch.float32,
                                         device=device)
    batch_target_values = convert_to_tensor(data=experience_buffers[7].copy(),
                                            data_type=torch.float32,
                                            device=device)

    for k in range(k_epoch):
        network.actor_optimizer.zero_grad()
        # [b, n, output_dim], [b, n, 24]
        multi_agent_policy, multi_agent_pred, _ = network.forward(batch_obs, None, batch_mask, num_meta=num_meta)
        multi_agent_pi_dist = Categorical(multi_agent_policy)
        # Calculate policy loss
        # [b, n]
        log_p = multi_agent_pi_dist.log_prob(batch_actions)
        # [b, n]
        imp_weights = torch.exp(log_p - batch_log_p_old)
        imp_weights = imp_weights[:-1, :]
        surr1 = imp_weights * batch_advantages
        surr2 = torch.clamp(imp_weights, 1.0 - eps_clip, 1.0 + eps_clip) * batch_advantages

        # [b-1, n] —> [1]
        policy_loss = -1 * torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        # [b, n] —> [1]
        entropy_loss = multi_agent_pi_dist.entropy()[:-1, :].mean()
        # [b, n, 24] -> [1]
        pred_loss = torch.sum(torch.square(multi_agent_pred.reshape(-1, 24) - batch_target_queue.reshape(-1, 24)),
                              dim=-1,
                              keepdim=True).mean()
        # Update actor network
        network.actor_optimizer.zero_grad()
        if model_type == 'SOCIALLIGHT':
            (policy_loss - el_factor * entropy_loss).backward()
        else:
            (policy_loss - el_factor * entropy_loss + pred_loss * ql_factor).backward()

        if grad_clip is not None:
            actor_norm = torch.nn.utils.clip_grad_norm_(network.actor_network.parameters(), grad_clip)
        else:
            actor_norm = get_gard_norm(network.actor_network.parameters())

        network.actor_optimizer.step()

        clip_frac = imp_weights.detach().gt(1 + eps_clip) | imp_weights.detach().lt(1 - eps_clip)
        clip_frac = convert_to_tensor(data=clip_frac, data_type=torch.float32, device=device).mean()

        p_l += convert_to_item(policy_loss)
        e_l += convert_to_item(entropy_loss)
        a_gn += convert_to_item(actor_norm)
        c_f += convert_to_item(clip_frac)
        a_p_l += convert_to_item(pred_loss)

    for k in range(k_epoch):
        network.critic_optimizer.zero_grad()
        if model_type == 'SOCIALLIGHT':
            # [b, n, output_dim]
            multi_agent_q_value, multi_agent_pred, _ = network.forward_v(batch_obs,
                                                                         batch_neighbor_a,
                                                                         [None, None],
                                                                         batch_mask,
                                                                         num_meta=num_meta)
            # [b, n, 1]
            multi_agent_value = multi_agent_q_value.gather(2, batch_actions.unsqueeze(-1))
        else:
            # [b, n, 1] and [b, n, 24]
            multi_agent_value, multi_agent_pred, _ = network.forward_v(batch_obs,
                                                                       batch_neighbor_a,
                                                                       [None, None],
                                                                       batch_mask,
                                                                       num_meta=num_meta)

        # Calculate value loss
        value_loss = torch.square(multi_agent_value.squeeze(-1)[:-1, :] - batch_target_values)
        value_loss = torch.mean(value_loss)
        # Prediction loss [b, n, 24] -> [1]
        pred_loss = torch.sum(torch.square(multi_agent_pred.reshape(-1, 24) - batch_target_queue.reshape(-1, 24)),
                              dim=-1,
                              keepdim=True).mean()
        # Backward
        network.critic_optimizer.zero_grad()
        if model_type == 'SOCIALLIGHT':
            (vl_factor * value_loss).backward()
        else:
            (vl_factor * value_loss + pred_loss * ql_factor).backward()

        # Update critic network
        if grad_clip is not None:
            critic_norm = torch.nn.utils.clip_grad_norm_(network.critic_network.parameters(), grad_clip)
        else:
            critic_norm = get_gard_norm(network.critic_network.parameters())

        network.critic_optimizer.step()

        v_l += convert_to_item(value_loss)
        c_gn += convert_to_item(critic_norm)
        c_p_l += convert_to_item(pred_loss)

    return p_l / k_epoch, v_l / k_epoch, e_l / k_epoch, a_gn / k_epoch, c_gn / k_epoch, c_f / k_epoch, a_p_l / k_epoch, c_p_l / k_epoch


def main():
    global_env = set_env(env_type=INPUT_PARAMS.ENV_TYPE, server_number=None)
    global_device = torch.device('cuda') if DEVICE_PARAMS.USE_GPU else torch.device('cpu')
    global_network = set_network(input_size=global_env.obs_space_n,
                                 output_size=global_env.action_space_n,
                                 agent_size=global_env.agent_space_n,
                                 actor_lr=NETWORK_PARAMS.A_LR_Q,
                                 critic_lr=NETWORK_PARAMS.C_LR_Q,
                                 device=global_device)

    if DEVICE_PARAMS.LOAD_MODEL:
        assert DEVICE_PARAMS.MODEL_PATH is not None
        print('====== Loading model ======\n')
        global_summary = SummaryWriter(DEVICE_PARAMS.MODEL_PATH + '/train_MATSC')
        checkpoint = torch.load(DEVICE_PARAMS.MODEL_PATH + '/model_MATSC/checkpoint.pkl')
        global_network.load_state_dict(checkpoint['model_state_dict'])
        global_network.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'][0])
        global_network.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'][1])
        curr_episode = checkpoint['epoch']
        torch.save(global_network.state_dict(),
                   DEVICE_PARAMS.MODEL_PATH + '/model_MATSC/state_dict_{}.pth'.format(curr_episode))
        print("====== Current episode set to {} ======\n".format(curr_episode))

    else:
        # Create New experiment directories
        print("====== Launching New Training ======\n")
        create_dirs(params=EXPERIMENT_PARAMS)
        global_summary = SummaryWriter(EXPERIMENT_PARAMS.TRAIN_PATH)
        create_config_json(path=EXPERIMENT_PARAMS.CONFIG_FILE_PATH,
                           params=create_config_dict())
        print('====== Logging Configuration ======\n')

        curr_episode = 0

    # launch all the threads:
    meta_agents = [Runner.remote(i) for i in range(DEVICE_PARAMS.NUM_META_AGENTS)]

    # get the initial weights from the global network
    weights = global_network.state_dict()
    if DEVICE_PARAMS.LOAD_MODEL:
        torch.save(weights, DEVICE_PARAMS.MODEL_PATH + '/model_MATSC/state_dict.pth')
    else:
        torch.save(weights, EXPERIMENT_PARAMS.MODEL_PATH + '/state_dict.pth')

    # launch the first job (e.g. getGradient) on each runner
    job_list = []  # Ray ObjectIDs
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(curr_episode))
    curr_episode += 1

    tensorboard_data = []
    try:
        while curr_episode < DEVICE_PARAMS.MAX_EPISODES:
            # wait for any job to be completed - unblock as soon as the earliest arrives
            done_id, job_list = ray.wait(job_list, num_returns=DEVICE_PARAMS.NUM_META_AGENTS)

            # get the results of the task from the object store
            # job_results, metrics, info = ray.get(done_id)[0]
            all_jobs = ray.get(done_id)
            global_buffer, global_metrics = get_global_train_buffer(all_jobs)

            if JOB_TYPE == JOB_OPTIONS.GET_EXPERIENCE:
                if NETWORK_PARAMS.UPDATE_TYPE == 'PPO_MAC':
                    train_metrics = calculate_gradients_ma_ppo(network=global_network,
                                                               device=global_device,
                                                               experience_buffers=global_buffer)


                else:
                    raise NotImplemented

                tensorboard_data.append(list(train_metrics) + list(global_metrics))

            else:
                print("Not implemented")
                assert (1 == 0)
            print("====== Finish updating the network ======")

            if not INPUT_PARAMS.EVAL_TRAFFIC:
                if len(tensorboard_data) >= DEVICE_PARAMS.SUMMARY_WINDOW:
                    write_to_Tensorboard(global_summary, tensorboard_data, curr_episode)
                    tensorboard_data = []
            else:
                write_to_Tensorboard_eval(global_summary, tensorboard_data, curr_episode)
                tensorboard_data = []

            # get the updated weights from the global network
            weights = global_network.state_dict()
            if DEVICE_PARAMS.LOAD_MODEL:
                torch.save(weights, DEVICE_PARAMS.MODEL_PATH + '/model_MATSC/state_dict.pth')
            else:
                torch.save(weights, EXPERIMENT_PARAMS.MODEL_PATH + '/state_dict.pth')
            curr_episode += 1

            # start a new job on the recently completed agent with the updated weights
            # job_list.extend([meta_agents[info["id"]].job.remote(curr_episode)])
            job_list = []
            for i, meta_agent in enumerate(meta_agents):
                job_list.append(meta_agent.job.remote(curr_episode))

            # Save model
            if curr_episode % DEVICE_PARAMS.SAVE_MODEL_STEP == 0:
                print('Saving Model!\n')
                checkpoint = {"model_state_dict": global_network.state_dict(),
                              "optimizer_state_dict": [global_network.actor_optimizer.state_dict(),
                                                       global_network.critic_optimizer.state_dict()],
                              "epoch": curr_episode}
                if DEVICE_PARAMS.LOAD_MODEL:
                    path_checkpoint = "./" + DEVICE_PARAMS.MODEL_PATH + "/model_MATSC/checkpoint.pkl"
                else:
                    path_checkpoint = "./" + EXPERIMENT_PARAMS.MODEL_PATH + "/checkpoint.pkl"

                torch.save(checkpoint, path_checkpoint)

            # Reset optimizer
            if NETWORK_PARAMS.RESET_OPTIM and curr_episode % NETWORK_PARAMS.RESET_OPTIM_STEP == 0:
                global_network.reset_optimizer()

            # Learning rate linear decay
            if NETWORK_PARAMS.LR_DECAY:
                update_linear_schedule(global_network.actor_optimizer,
                                       curr_episode,
                                       INPUT_PARAMS.MAX_EPISODES,
                                       global_network.actor_lr)
                update_linear_schedule(global_network.critic_optimizer,
                                       curr_episode,
                                       INPUT_PARAMS.MAX_EPISODES,
                                       global_network.critic_lr)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":

    seed = 0
    # Set random number generator
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
