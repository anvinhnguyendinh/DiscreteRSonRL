import torch
import numpy as np


def evaluate_attack(model, attacker, env, num_frames, per_action=True, num_episodes=1, saved_shape=None):
    all_epi_len = []
    all_epi_obs = []

    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        all_obs, count = np.zeros((num_frames, saved_shape)), 0
        while not done:

            action0, states = model.predict(obs [0], deterministic=True)
            # obs = np.expand_dims(obs, 0)

            attack_flag, obs_ = attacker.attack(obs, count)
            action, _states = model.predict(obs_[0], deterministic=True)
            # print(action0, action, model.policy.action_net(model.policy._get_latent(torch.FloatTensor(obs))[0]), model.policy.action_net(model.policy._get_latent(torch.FloatTensor(obs_))[0]))
            cert_flag = float(np.array((action0 == action) * 1.0))
            action1 = action if per_action else action0

            all_obs[count] = list(obs[0]) + list(obs_[0]) + [action1] #
            obs, reward, done, info = env.step(np.array([action1]))
            episode_rewards.append([reward[0], cert_flag])
            count += 1

        all_episode_rewards.append(list(np.mean(episode_rewards, axis=0)) + [count])
        all_episode_rewards[-1][0]  *=  count
        all_epi_obs.append(all_obs)
        all_epi_len.append(count)

    return all_episode_rewards, all_epi_obs, all_epi_len


def evaluate_smooth(model_sm, attacker, env, num_frames, steps=1, per_action=True, mode=0, num_episodes=1, saved_shape=None):
    all_epi_len = []
    all_epi_obs = []
    upper_bound = 1e-8 * model_sm.sigma #
    lower_bound = attacker.eps #* np.sqrt(steps)

    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        all_obs, count = np.zeros((num_frames, saved_shape)), 0
        while not done:

            action0, rad0, actionb = model_sm.certify_mode(obs , 2*i  , count % steps)
            action0 = actionb[mode] # non m = 2 | no-a action = actionb[2]

            # obs = np.expand_dims(obs, 0)
            attack_flag, obs_ = attacker.attack(obs, count)
            # print(count, np.linalg.norm(attacker.perturbation /  attacker.scaler))
            # obs = obs_[0]

            action , rad1, actionc = model_sm.certify_mode(obs_, 2*i+1, count % steps) #

            cert_flag = (action0 == action )
            cert2flag = (action0 == actionc[mode])

            if per_action[0]:
                action1 = per_action[1](actionc[mode])
            else:
                action1 = per_action[1](action0)

            all_obs[count] = list(obs[0]) + list(obs_[0]) + [np.array(action1)] + [rad0, rad1 * (2 * cert_flag  - 1)] #

            obs, reward, done, info = env.step(np.array([action1]))
            episode_rewards.append([reward[0], cert_flag, cert2flag, rad0 > lower_bound, \
                rad0 < upper_bound, rad1 > lower_bound, rad1 < upper_bound, rad0 - rad1 * (2 * cert_flag - 1), \
                action0 == actionc[2], action == actionc[2], rad1 == 0 and  action0 == actionc[2],  action0 == actionb[2]]) #

            count += 1

        episode_rewards = np.mean (episode_rewards, axis=0)
        episode_rewards = list(episode_rewards) + list(np.max(all_obs[:, -2:], axis=0) / model_sm.sigma) + [count]

        all_episode_rewards.append(episode_rewards)
        all_episode_rewards[-1][0]  *=  count
        all_epi_obs.append(all_obs)
        all_epi_len.append(count)

    return all_episode_rewards, all_epi_obs, all_epi_len


def evaluate_reward(model_sm, attacker, env, num_frames, steps=1, per_action=True, mode=0, num_episodes=1, saved_shape=None):
    all_epi_len = []
    all_epi_obs = []
    upper_bound = 1e-8 * model_sm.sigma
    lower_bound = attacker.eps #* np.sqrt(steps)

    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        all_obs, count, sum_rewards = np.zeros((num_frames, saved_shape)), 0, 0 #
        while not done:

            action0, rad0, actionb, reward0 = model_sm.certify_mode(obs , 2*i  , count % steps)
            action0 = actionb[mode]

            # obs = np.expand_dims(obs, 0)
            attack_flag, obs_ = attacker.attack(obs, count)
            # obs = obs_[0]

            action , rad1, actionc, reward1 = model_sm.certify_mode(obs_, 2*i+1, count % steps)

            cert_flag = (reward0 == reward1)
            cert1flag = (action0 == action )
            cert2flag = (action0 == actionc[mode])

            if per_action[0]:
                action1 = per_action[1](actionc[mode])
            else:
                action1 = per_action[1](action0)

            all_obs[count] = list(obs[0]) + list(obs_[0]) + [np.array(action1)] + [reward0, reward1, 0, rad0, rad1 * (2 * cert_flag -1)]

            obs, reward, done, info = env.step(np.array([action1]))
            all_obs[count, -3] = reward[0]

            sum_rewards += reward[0]

            if (count + 1) % steps == 0:
                all_obs[count, -3] = sum_rewards
                sum_rewards = 0

            episode_rewards.append([reward[0], cert1flag, cert2flag, rad0 > lower_bound, \
                rad0 < upper_bound, rad1 > lower_bound, rad1 < upper_bound, rad0 - rad1 * (2 * cert_flag - 1), \
                (action0 == actionc[2]), action == actionc[2], rad1 == 0 and  action0 == actionc[2],  action0 == actionb[2], cert_flag])

            count += 1

        episode_rewards = np.mean (episode_rewards, axis=0)
        episode_rewards = list(episode_rewards) + list(np.max(all_obs[:, -2:-1], axis=0) / model_sm.sigma)  + [count]

        all_episode_rewards.append(episode_rewards)
        all_episode_rewards[-1][0]  *=  count
        all_epi_obs.append(all_obs)
        all_epi_len.append(count)

    return all_episode_rewards, all_epi_obs, all_epi_len