#@title OptimizedParallelAtariEnv
import cv2
import numpy as np
from collections import deque
import gymnasium as gym

class OptimizedParallelAtariEnv:
    """学習効率を向上させた並列Atari Breakout環境"""

    def __init__(self, env_id, n_env, n_stack=4, seed=42, frame_skip=4, noop_max=30):
        self.n_env = n_env
        self.n_stack = n_stack
        self.frame_skip = frame_skip  # フレームスキップ数
        self.noop_max = noop_max  # 初期のランダムアクション数
        self.envs = []
        self.frame_stacks = []
        self.raw_frames = []
        self.lives = []  # ライフ数を追跡
        
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
            env = gym.make(working_env, render_mode='rgb_array')
            env.reset(seed=seed + i)
            self.envs.append(env)
            self.frame_stacks.append(deque(maxlen=n_stack))
            self.raw_frames.append(None)
            self.lives.append(0)

    def _preprocess_frame(self, frame):
        """
        最適化されたAtari前処理:
        1. RGB → グレースケール変換（効率的な重み付き変換）
        2. 84x84リサイズ
        3. [0,1]正規化
        """
        if len(frame.shape) == 3:
            # より効率的なグレースケール変換（OpenCVより高速）
            # Y = 0.299*R + 0.587*G + 0.114*B の重み付き変換
            gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
            gray = gray.astype(np.uint8)
        else:
            gray = frame
        
        # リサイズ（INTER_AREAは縮小時に最適）
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # [0, 1]に正規化（学習の安定化）
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized

    def _get_max_frame(self, frame1, frame2):
        """フリッカー対策：2フレームの最大値を取る"""
        return np.maximum(frame1, frame2)

    def reset(self):
        """リセット（ランダムなnoop actionを実行）"""
        obs_list = []

        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            self.raw_frames[i] = obs.copy()
            
            # ライフ数を記録
            self.lives[i] = info.get('lives', 0)
            
            # ランダムなnoop actionを実行（ゲーム開始の多様性確保）
            noop_steps = np.random.randint(1, self.noop_max + 1)
            for _ in range(noop_steps):
                obs, _, terminated, truncated, info = env.step(0)  # 0 = noop action
                if terminated or truncated:
                    obs, info = env.reset()
                    break
            
            self.raw_frames[i] = obs.copy()
            processed = self._preprocess_frame(obs)

            # フレームスタックを初期化
            self.frame_stacks[i].clear()
            for _ in range(self.n_stack):
                self.frame_stacks[i].append(processed)

            stacked = np.stack(list(self.frame_stacks[i]), axis=-1)
            obs_list.append(stacked)

        return np.array(obs_list)

    def step(self, actions):
        """
        最適化されたステップ:
        1. フレームスキップによる高速化
        2. 報酬クリッピング
        3. ライフロス検出
        """
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

            # フレームスキップ実行
            total_reward = 0.0
            last_frame = None
            current_frame = None
            
            for skip in range(self.frame_skip):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # フリッカー対策：最後の2フレームを保存
                if skip == self.frame_skip - 2:
                    last_frame = obs.copy()
                elif skip == self.frame_skip - 1:
                    current_frame = obs.copy()
                
                if terminated or truncated:
                    break
            
            # フリッカー対策：2フレームの最大値を取る
            if last_frame is not None and current_frame is not None:
                obs = self._get_max_frame(last_frame, current_frame)
            else:
                obs = current_frame if current_frame is not None else obs

            done = terminated or truncated
            
            # ライフロス検出（Breakoutの場合）
            life_lost = False
            if 'lives' in info:
                if info['lives'] < self.lives[i] and info['lives'] > 0:
                    life_lost = True
                self.lives[i] = info['lives']

            self.raw_frames[i] = obs.copy()

            # 前処理
            processed = self._preprocess_frame(obs)
            self.frame_stacks[i].append(processed)

            # エピソード終了時はリセット
            if done:
                obs, info = env.reset()
                self.lives[i] = info.get('lives', 0)
                self.raw_frames[i] = obs.copy()
                processed = self._preprocess_frame(obs)
                self.frame_stacks[i].clear()
                for _ in range(self.n_stack):
                    self.frame_stacks[i].append(processed)

            # 報酬クリッピング（学習の安定化）
            clipped_reward = np.sign(total_reward)

            stacked = np.stack(list(self.frame_stacks[i]), axis=-1)
            obs_list.append(stacked)
            rewards.append(clipped_reward)
            dones.append(done)
            
            # 追加情報
            info['life_lost'] = life_lost
            info['raw_reward'] = total_reward
            infos.append(info)

        return np.array(obs_list), np.array(rewards), np.array(dones), infos

    def get_images(self):
        """元画像を取得（動画作成用）"""
        return [frame.copy() for frame in self.raw_frames]
    
    def get_action_space(self):
        """アクション空間を取得"""
        return self.envs[0].action_space
    
    def get_observation_space(self):
        """観測空間を取得"""
        return (84, 84, self.n_stack)
    
    def close(self):
        """環境を閉じる"""
        for env in self.envs:
            env.close()
    
    def render(self, env_idx=0):
        """指定した環境の画面を表示"""
        if 0 <= env_idx < self.n_env:
            return self.raw_frames[env_idx]
        return None

# 使用例とテスト用のヘルパー関数
def test_environment():
    """環境のテスト用関数"""
    env = OptimizedParallelAtariEnv(
        env_id='Breakout',
        n_env=4,
        n_stack=4,
        seed=42,
        frame_skip=4
    )
    
    print(f"アクション空間: {env.get_action_space()}")
    print(f"観測空間: {env.get_observation_space()}")
    
    # 初期化
    obs = env.reset()
    print(f"初期観測形状: {obs.shape}")
    print(f"観測値の範囲: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # ランダムアクションでテスト
    for step in range(10):
        actions = [env.get_action_space().sample() for _ in range(4)]
        obs, rewards, dones, infos = env.step(actions)
        print(f"ステップ {step+1}: 報酬={rewards}, 終了={dones}")
        
        if any(dones):
            print("エピソード終了を検出")
    
    env.close()
    print("テスト完了")

if __name__ == "__main__":
    test_environment()
