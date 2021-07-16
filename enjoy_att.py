import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict

import time
from randsm.attack import SimpleAttack
from randsm.agent import set_randsm_seed
from randsm.custom import ENV_CUSTOM_INFO
from randsm.smooth import Smooth, SmoothMultiA, SmoothReward, SmoothMultiR
from randsm.evaluate import evaluate_attack, evaluate_smooth, evaluate_reward


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=150000, type=int) #

    parser.add_argument("--n-seeds", help="number of seeds", default=30, type=int)
    parser.add_argument("--att-folder", help="result folder", type=str, default="att_logs")
    parser.add_argument("--n-episodes", help="number of episodes per seed", default=10, type=int)
    parser.add_argument("--attack-clean", action="store_true", default=False, help="Use attack eval: Clean")

    parser.add_argument("--n-actions", help="number of discrete actions", default=2, type=int)
    parser.add_argument("--n-rewards", help="number of discrete rewards", default=2, type=int)
    parser.add_argument("--n-steps", help="number of smooth steps in Multi-RS ", default=1, type=int)
    parser.add_argument("--batch-size", help="size of a batch in smoothed agents", default=50, type=int) #
    parser.add_argument("--attack-smooth", action="store_true", default=False, help="Use attack eval: Smooth ")
    parser.add_argument("--attack-smooth-mode", help="Use attack eval: Smooth type and mode", default=0, type=int) #
    parser.add_argument("--attack-smooth-reward", action="store_true", default=False, help="Use Smooth type: Reward RS ")

    parser.add_argument("--eps", help="attack param: eps", default=0.09, type=float)
    parser.add_argument("--nb-iter", help="attack param: nb_iter", default=91, type=int)
    parser.add_argument("--eps-iter", help="attack param: eps_iter", default=0.01, type=float)
    parser.add_argument("--sigma", help="sigma of smoothed agents", default=0.0, type=float) #
    parser.add_argument("--attack-perturb", action="store_true", default=False, help="Use attack mode: Perturb") #
    parser.add_argument("--attack-previous", action="store_true", default=False, help="Use attack mode: Previous") #
    parser.add_argument("--attack-rand-init", action="store_true", default=False, help="Use attack mode: Rand init") #


    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    obs = env.reset() #

    init_flag = args.attack_rand_init
    pre_flag = not args.attack_perturb
    pos_flag = 1 - args.attack_previous
    unclean_flag = not args.attack_clean
    env_add_info = ENV_CUSTOM_INFO[env_id]
    smooth_mode = args.attack_smooth_mode % 3
    continuous_flag = (env_add_info.CLAS == 0)
    np.set_printoptions(suppress=True, precision=8)
    smooth_flag   = args.attack_smooth and args.sigma > 0 #
    smooth_reward = smooth_flag and args.attack_smooth_reward
    res_path = os.path.join(args.att_folder, args.algo, f"{env_id}")

    if pos_flag == 1:
        pre_flag = True
    if env_add_info.CLAS > 1:
        args.n_actions = env_add_info.CLAS
    elif smooth_reward:
        args.n_actions = 0
    if env_add_info.RECL > 1 and args.n_steps <= 1:
        args.n_rewards = env_add_info.RECL

    name = '_'.join(map(str, [unclean_flag, '%.02f'% args.eps,  '|', args.seed]))
    if not smooth_flag:
        name = '_'.join(map(str, ['data_attack', name]))
    elif not smooth_reward:
        name = '_'.join(map(str, ['data_smooth', args.n_actions, args.n_steps, '|', smooth_mode, '%.02f'% args.sigma, '|', name]))
    else:
        name = '_'.join(map(str, ['data_reward', args.n_rewards, args.n_steps, '|', smooth_mode, '%.02f'% args.sigma, '|', name]))


    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []

    try: #
        dict_flag = smooth_flag + smooth_reward
        episodes_frames = args.n_episodes *env_add_info.FRAM
        args.n_seeds = args.n_timesteps // episodes_frames
        args.n_timesteps = args.n_seeds *  episodes_frames
        data_save_0 = np.zeros((args.n_seeds, args.n_episodes, env_add_info.FRAM, env_add_info.DICT[dict_flag][0]))
        data_save_1 = np.zeros((args.n_seeds, args.n_episodes, env_add_info.DICT[ dict_flag][1]))
        env_add_info.eval(algo)

        if smooth_flag:
            if not smooth_reward:
                if args.n_steps <= 1:
                    smoothed_agent = Smooth      (model, args.n_actions, args.sigma, env_add_info.SCAL,                                  args.batch_size, env_add_info.ACSP, continuous_flag)
                else:
                    smoothed_agent = SmoothMultiA(model, args.n_actions, args.sigma, env_add_info.SCAL, env_add_info.TENV, args.n_steps, args.batch_size, env_add_info.ACSP, continuous_flag)
            else:
                if args.n_steps <= 1:
                    smoothed_agent = SmoothReward(model, args.n_rewards, args.sigma, env_add_info.SCAL, env_add_info.TENV, args.n_steps, args.batch_size, env_add_info.ACSP, args.n_actions, env_add_info.RESP)
                else:
                    smoothed_agent = SmoothMultiR(model, args.n_rewards, args.sigma, env_add_info.SCAL, env_add_info.TENV, args.n_steps, args.batch_size, env_add_info.ACSP, args.n_actions, env_add_info.RESP)
            smoothed_agent.mode  = args.attack_smooth_mode // 3
            smoothed_agent.is_hat  = smooth_mode == 2
            if smooth_mode > 1:
                smooth_mode %= 2

        attack_agent = SimpleAttack(model, pre_flag, pos_flag, args.eps , args.nb_iter , rand_init=init_flag, clip=env_add_info.LIMI, \
                                                        scaler=env_add_info.SCAL, batch_size=args.batch_size, loss=continuous_flag  )
        start = time.time()

        for time_count in range(args.n_timesteps):
            if time_count % episodes_frames != 0:
                continue

            seed_incre = time_count // episodes_frames
            set_randsm_seed(args.seed + seed_incre, env)

            if smooth_flag:
                smoothed_agent.master_seed(args.seed + seed_incre, args.n_episodes *2)
                unclean_flag2 = (unclean_flag, lambda x: [x]) if continuous_flag else (unclean_flag, lambda x: x)

                if smooth_reward:
                    mr, data, ml = evaluate_reward(smoothed_agent, attack_agent, env, env_add_info.FRAM, args.n_steps, unclean_flag2, smooth_mode, args.n_episodes, data_save_0.shape[-1])
                else:
                    mr, data, ml = evaluate_smooth(smoothed_agent, attack_agent, env, env_add_info.FRAM, args.n_steps, unclean_flag2, smooth_mode, args.n_episodes, data_save_0.shape[-1])
            else:
                mr,     data, ml = evaluate_attack(         model, attack_agent, env, env_add_info.FRAM,               unclean_flag ,              args.n_episodes, data_save_0.shape[-1])

            data_save_0[seed_incre] = data
            data_save_1[seed_incre] = mr
            infos, done = [{}], True

            # action, state = model.predict(obs, state=state, deterministic=deterministic)
            # obs, reward, done, infos = env.step(action)
            if not args.no_render:
                env.render("human")

            episode_reward = np.mean(mr, axis=0)
            ep_len = np.mean(ml, axis=0)

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    # print(f"Episode Reward: {episode_reward}, std {np.std(mr, axis=0)}") #
                    # print(f"Episode Length: {ep_len}, std {np.std(ml, axis=0)}")
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0
                    state = None


                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        # print(f"{len(episode_rewards) * args.n_episodes} Episodes") #
        if args.attack_smooth_reward:
            args.n_actions = args.n_rewards
        print(' '.join(map(str,['Reward RS:', args.attack_smooth_reward, args.n_actions, args.n_steps])))
        print(' '.join(map(str,[args.eps, args.attack_smooth_mode, args.sigma, unclean_flag, args.n_steps])))
        print(f"Mean reward: {np.mean(episode_rewards, axis=0)} +/- {np.std(episode_rewards, axis=0)}"  )

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths, axis=0)} +/- {np.std(episode_lengths, axis=0)}")

    # print(f"Seed root: {args.seed}")
    # print(f"Time (ms): {1000 * (time.time() - start) / args.n_episodes / (sum(episode_lengths) + 1e-7)}")
    env.close()

    np.save(os.path.join(res_path, f"{name}.npy"), {0: data_save_0, 1: data_save_1})



if __name__ == "__main__":
    main()
