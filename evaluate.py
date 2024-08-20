import argparse
import os.path

import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
'''
use tensorboard  :
tensorboard --logdir=/home/user/thu_ee/maddpg-pytorch/models/simple_tag/target_hunting_4/run1/logs
'''
def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)#incremental是增量，逐步的意思
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        print('this is config.save_gifs',config.save_gifs)
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)
        print(gif_path)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    #保存数据 TODO 优化这个代码的结构
    obs_files = ["obs_0.npy","obs_1.npy","obs_2.npy"]
    obs_data = {file: [] for file in obs_files}
    ac_files = ["acs_0.npy", "acs_1.npy", "acs_2.npy"]
    ac_data = {file: [] for file in ac_files}
    done_files = ["dones_0.npy", "dones_1.npy", "dones_2.npy"]
    done_data = {file: [] for file in done_files}
    re_files = ["rews_0.npy", "rews_1.npy", "rews_2.npy"]
    re_data = {file: [] for file in re_files}

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames = []
            #print(env.render('rgb_array'))
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # convert action to numpy arrays
            np_obs = [ob.data.numpy().flatten() for ob in torch_obs]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)

            for i in range(3):
                obs_data[obs_files[i]].append(np_obs[i])
                ac_data[ac_files[i]].append(actions[i])
                done_data[done_files[i]].append(dones[i])
                re_data[re_files[i]].append(rewards[i])

            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

    env.close()
    data_path = 'E:/postgraduate/maddpg_fix_agent/dataset/seed_9_data'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for file in obs_files:
        np.save(os.path.join(data_path,file),np.array(obs_data[file]))
    for file in ac_files:
        np.save(os.path.join(data_path,file),np.array(ac_data[file]))
    for file in re_files:
        np.save(os.path.join(data_path,file),np.array(re_data[file]))
    for file in done_files:
        np.save(os.path.join(data_path,file),np.array(done_data[file]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",default = 'simple_tag', help="Name of environment")
    parser.add_argument("--model_name",default = 'fix_agent', help="Name of model")
    parser.add_argument("--run_num", default=1, type=int)
    parser.add_argument("--save_gifs", default = False , action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=40001, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")#这个参数是要改的
    parser.add_argument("--n_episodes", default=1000, type=int)
    parser.add_argument("--episode_length", default=40, type=int)##important
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)