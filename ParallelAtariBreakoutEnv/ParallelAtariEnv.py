# @title ParallelAtariEnv

import numpy as np
from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray
import gym

class ParallelAtariEnv:
    def __init__(self, env_id, n_envs, n_stack, seed=None):
        self.envs = [gym.make(env_id, frameskip=4) for _ in range(n_envs)]
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.stacked_frames = [deque(maxlen=n_stack) for _ in range(n_envs)]
        self.action_meanings = self.envs[0].unwrapped.get_action_meanings()  # 行動の意味を取得
        self.last_reset = [True] * n_envs  # 各環境のリセット直後フラグ

        if seed is not None:
            for i, env in enumerate(self.envs):
                env.seed(seed + i)

    def preprocess_frame(self, frame):
        # グレースケール変換 + リサイズ (84x84)
        frame = frame[8:194, :] # darray (210, 160, 3) から必要な部分のみ切り出す
        gray_frame = rgb2gray(frame)
        resized_frame = resize(gray_frame, (84, 84), anti_aliasing=True)  # -> (84, 84)
        return resized_frame.astype(np.float32)

    def is_ball_missing(self, stacked_state):
        """
        ボールが消失しているか判定する関数。

        Args:
            stacked_state: dequeに格納されたフレーム（リスト形式）

        Returns:
            True: ボールが消失している
            False: ボールが存在している
        """
        # スタックされた状態をnumpy配列に変換
        stacked_array = np.stack(list(stacked_state), axis=-1)  # (84, 84, n_stack)

        # パドルより上の部分を抽出
        upper_limit = 74 # 74 行目からpaddle画像がある
        upper_area = stacked_array[:upper_limit, :, :]  # 上部の領域

        # 上部領域がスタック全体で変化しないか判定(stackの最初と最後を比較)
        return np.all(upper_area[..., 0] == upper_area[..., N_STACK - 1])

    def reset(self):
        states = []
        for i, env in enumerate(self.envs):
            state = env.reset()
            state = self.preprocess_frame(state)  # 1フレームを前処理
            for _ in range(self.n_stack):
                self.stacked_frames[i].append(state)
            states.append(np.stack(self.stacked_frames[i], axis=-1))  # (84, 84, n_stack)
            self.last_reset[i] = True  # リセット直後フラグをセット
        return np.array(states)  # (n_envs, 84, 84, n_stack)

    def step(self, actions):
        # アクションを修正：ボール消失時にFIREを自動実行
        for i, stacked_state in enumerate(self.stacked_frames):
            if self.is_ball_missing(stacked_state):
                actions[i] = self.action_meanings.index("FIRE")

        states, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done, info = env.step(action)
            state = self.preprocess_frame(state)  # フレームを前処理

            if done:
                state = env.reset()
                state = self.preprocess_frame(state)
                self.stacked_frames[i].clear()
                for _ in range(self.n_stack):
                    self.stacked_frames[i].append(state)
                self.last_reset[i] = True  # リセット直後フラグを再設定
            else:
                self.stacked_frames[i].append(state)

            states.append(np.stack(self.stacked_frames[i], axis=-1))  # (84, 84, n_stack)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return np.array(states), np.array(rewards), np.array(dones), infos

    def get_compressed_image(self, state):
        _state = self.preprocess_frame(state)  # フレームを前処理
        im_state = np.array([_state] * 3)
        im_state = im_state.transpose(1, 2, 0)
        im_state = (im_state * 255).astype(np.uint8)
        return Image.fromarray(im_state)

    def get_images(self):
        """
        現在の各環境の元サイズ（カラー画像）を取得する関数。

        Returns:
            images (list of np.array): 各環境の現在の観測データ (210x160x3) のリスト
        """
        images = []
        for env in self.envs:
            if hasattr(env.unwrapped, "render"):
                image = env.unwrapped.render(mode="rgb_array")  # 210x160x3の画像を取得
                images.append(image)
            else:
                raise RuntimeError("The environment does not support rendering.")
        return images