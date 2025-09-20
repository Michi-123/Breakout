#!pip install gymnasium ale-py opencv-python torch

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from collections import deque
import random

import ale_py

# 動作する環境を見つける
def find_working_env():
    env_candidates = [
        # 'ALE/Breakout-v5',
        # 'Breakout-v4',
        'BreakoutNoFrameskip-v4'
    ]

    for env_name in env_candidates:
        try:
            env = gym.make(env_name, render_mode=None)
            obs, _ = env.reset()
            env.close()
            print(f"使用可能な環境: {env_name}")
            return env_name
        except Exception as e:
            print(f"{env_name} 不可: {str(e)[:50]}...")

    raise Exception("動作する環境が見つかりません")

ENV_NAME = find_working_env()
