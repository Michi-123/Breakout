#@title SimpleParallelAtariEnv
import gym
import cv2
import numpy as np
from collections import deque

class SimpleParallelAtariEnv:
    """ParallelAtariBreakoutEnvの代替（元のコードと完全互換）"""

    def __init__(self, env_id, n_env, n_stack, seed):
        self.n_env = n_env
        self.n_stack = n_stack
        self.envs = []
        self.frame_stacks = []
        self.raw_frames = []  # 元画像保存用

        # 動作する環境名を見つける
        env_candidates = ['ALE/Breakout-v5', 'Breakout-v4', 'BreakoutNoFrameskip-v4']
        working_env = None

        for env_name in env_candidates:
            try:
                test_env = gym.make(env_name, render_mode=None)
                test_env.reset(seed=seed)
                test_env.close()
                working_env = env_name
                break
            except:
                continue

        if not working_env:
            raise RuntimeError("動作する環境が見つかりません")

        print(f"使用環境: {working_env}")

        # 環境を初期化
        for i in range(n_env):
            env = gym.make(working_env, render_mode='rgb_array')  # rgb_arrayモードに変更
            env.reset(seed=seed + i)
            self.envs.append(env)
            self.frame_stacks.append(deque(maxlen=n_stack))
            self.raw_frames.append(None)  # 最新の元画像を保存

    def _preprocess_frame(self, frame):
        """Atari前処理: グレースケール + 84x84リサイズ"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self):
        """リセット（元のコードと同じ戻り値形式）"""
        obs_list = []

        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self.raw_frames[i] = obs.copy()  # 元画像を保存
            processed = self._preprocess_frame(obs)

            # フレームスタックを初期化
            self.frame_stacks[i].clear()
            for _ in range(self.n_stack):
                self.frame_stacks[i].append(processed)

            # スタックして追加
            stacked = np.stack(list(self.frame_stacks[i]), axis=-1)
            obs_list.append(stacked)

        return np.array(obs_list)

    def step(self, actions):
        """ステップ（元のコードと同じ戻り値形式）"""
        obs_list = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # アクション変換
            if isinstance(action, (list, np.ndarray)):
                action = int(action[0])
            else:
                action = int(action)

            # ステップ実行
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.raw_frames[i] = obs.copy()  # 元画像を保存

            # 前処理
            processed = self._preprocess_frame(obs)
            self.frame_stacks[i].append(processed)

            # エピソード終了時はリセット
            if done:
                obs, _ = env.reset()
                self.raw_frames[i] = obs.copy()  # リセット後の画像も保存
                processed = self._preprocess_frame(obs)
                self.frame_stacks[i].clear()
                for _ in range(self.n_stack):
                    self.frame_stacks[i].append(processed)

            # スタックして追加
            stacked = np.stack(list(self.frame_stacks[i]), axis=-1)
            obs_list.append(stacked)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        # 元のコードと同じ形式で戻り値を返す
        return np.array(obs_list), np.array(rewards), np.array(dones), infos

    def get_images(self):
        """元画像を取得（動画作成用）"""
        return [frame.copy() for frame in self.raw_frames]
