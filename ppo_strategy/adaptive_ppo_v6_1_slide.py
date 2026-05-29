import pandas as pd
import numpy as np
import os
import glob
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# =======================================================
# 1. 核心底层：Numpy 向量化缓存系统 (复用)
# =======================================================
def load_numpy_cache(parquet_dir, macro_csv):
    print("Loading macro & Parquet data into Numpy Tensors...")
    macro_df = pd.read_csv(macro_csv)
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.set_index('date', inplace=True)
    
    macro_cols = ['limit_down_rate', 'limit_up_rate', 'up_down_ratio', 'market_breadth',
                  'alpha_divergence', 'idx_bias_20', 'idx_vol_ratio_5', 'idx_return']
    
    rolling_mean = macro_df[macro_cols].rolling(window=250, min_periods=20).mean()
    rolling_std = macro_df[macro_cols].rolling(window=250, min_periods=20).std()
    expanding_mean = macro_df[macro_cols].expanding(min_periods=1).mean()
    expanding_std = macro_df[macro_cols].expanding(min_periods=1).std()
    
    rolling_mean = rolling_mean.fillna(expanding_mean).fillna(0)
    rolling_std = rolling_std.fillna(expanding_std).fillna(1e-8)
    rolling_std[rolling_std == 0] = 1e-8
    
    macro_z = (macro_df[macro_cols] - rolling_mean) / rolling_std
    macro_z = macro_z.clip(-3.0, 3.0)
    macro_z.index = macro_z.index.strftime('%Y-%m-%d')
    macro_dict = macro_z.to_dict(orient='index')

    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    valid_dates = []
    stock_to_idx = {}
    
    for f in files:
        date_str = os.path.basename(f).split(".")[0]
        if date_str in macro_dict:
            valid_dates.append(date_str)
            df = pd.read_parquet(f, columns=[]) 
            for code in df.index:
                if code not in stock_to_idx:
                    stock_to_idx[code] = len(stock_to_idx)
    
    num_days = len(valid_dates)
    num_stocks = len(stock_to_idx)
    print(f"Total days: {num_days}, Total unique stocks: {num_stocks}")
    
    rank_matrix = np.full((num_days, num_stocks), 1.0, dtype=np.float32) 
    return_matrix = np.zeros((num_days, num_stocks), dtype=np.float32)
    trade_mask = np.zeros((num_days, num_stocks), dtype=bool) 
    sell_lock_mask = np.zeros((num_days, num_stocks), dtype=bool) 
    macro_matrix = np.zeros((num_days, 8), dtype=np.float32)

    for i, date_str in enumerate(valid_dates):
        f = os.path.join(parquet_dir, f"{date_str}.parquet")
        df = pd.read_parquet(f)
        indices = [stock_to_idx[c] for c in df.index]
        
        rank_matrix[i, indices] = df['rank_pct'].values
        return_matrix[i, indices] = df['raw_return'].values
        
        can_buy = (~df['is_limit_up']) & (~df['is_halt']) & (~df['is_st'])
        trade_mask[i, indices] = can_buy.values
        
        cant_sell = df['is_limit_down'] | df['is_halt']
        sell_lock_mask[i, indices] = cant_sell.values
        
        m = macro_dict[date_str]
        macro_matrix[i] = [
            m['limit_down_rate'], m['limit_up_rate'], m['up_down_ratio'],
            m['market_breadth'], m['alpha_divergence'], m['idx_bias_20'],
            m['idx_vol_ratio_5'], m['idx_return']
        ]
        
    train_idx = [i for i, d in enumerate(valid_dates) if d <= "2025-06-30"]
    train_dates = [valid_dates[i] for i in train_idx]
    
    cache = {
        'rank': rank_matrix,
        'return': return_matrix,
        'trade': trade_mask,
        'sell_lock': sell_lock_mask,
        'macro': macro_matrix,
        'num_stocks': num_stocks
    }
    
    return train_idx, train_dates, cache

# =======================================================
# 2. V6-1 环境: 1D动作 单维度滑窗选股 (Sliding Window)
# =======================================================
class RLTraderEnvV6_1(gym.Env):
    def __init__(self, step_indices, dates, cache):
        super(RLTraderEnvV6_1, self).__init__()
        self.step_indices = step_indices
        self.dates = dates
        self.cache = cache
        
        self.INITIAL_CAPITAL = 100_000_000.0
        self.SELL_RANK_PCT = 0.4
        self.BUY_COST_RATE = 0.0008
        self.SELL_COST_RATE = 0.0018
        self.WINDOW_WIDTH = 150 # 固定的买入宽度 150 只票
        
        # 动作空间：1维连续值，决定宽度为150的滑窗的起点 [0, 1] 映射到 [0, 300]名
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # 观察空间：8维宏观 + 回撤水位 + 胜率 = 10维
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
        
    def _get_state(self):
        global_idx = self.step_indices[self.current_step]
        macro_obs = self.cache['macro'][global_idx]
        
        dd = (self.total_asset - self.peak_asset) / self.peak_asset if self.peak_asset > 0 else 0.0
        dd_z = np.clip(dd * 10.0, -3.0, 3.0) 
        
        held_count = np.sum(self.holdings > 0)
        held_z = np.clip((held_count - 150) / 100.0, -3.0, 3.0) # 中心对齐 150
        
        obs = np.append(macro_obs, [dd_z, held_z]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.INITIAL_CAPITAL
        self.holdings = np.zeros(self.cache['num_stocks'], dtype=np.float32)
        self.total_asset = self.INITIAL_CAPITAL
        self.peak_asset = self.INITIAL_CAPITAL
        return self._get_state(), {}

    def step(self, action):
        # AI 输出1个动作，决定滑窗起始点 (映射到 0~300)
        val = np.clip(action[0], 0.0, 1.0)
        start_rank = int(val * 300)
        end_rank = start_rank + self.WINDOW_WIDTH
        
        global_idx = self.step_indices[self.current_step]
        
        t_ranks = self.cache['rank'][global_idx]
        t_returns = self.cache['return'][global_idx]
        t_trade = self.cache['trade'][global_idx]
        t_sell_lock = self.cache['sell_lock'][global_idx]
        
        # ================================
        # 1. 向量化卖出逻辑 (固定规则)
        # ================================
        has_pos_mask = (self.holdings > 0)
        need_sell_mask = has_pos_mask & (t_ranks > self.SELL_RANK_PCT) & (~t_sell_lock)
        
        if np.any(need_sell_mask):
            sold_val = np.sum(self.holdings[need_sell_mask])
            self.cash += sold_val * (1 - self.SELL_COST_RATE)
            self.holdings[need_sell_mask] = 0.0
            has_pos_mask = (self.holdings > 0)
            
        # ================================
        # 2. 向量化买入逻辑 (单维滑窗)
        # ================================
        if self.cash > 0:
            cand_mask = t_trade & (~has_pos_mask)
            valid_indices = np.where(cand_mask)[0]
            
            if len(valid_indices) > 0:
                valid_ranks = t_ranks[valid_indices]
                sorted_relative_idx = np.argsort(valid_ranks)
                
                # 滑窗切片 [Start, Start + 150]
                buys_relative_idx = sorted_relative_idx[start_rank:end_rank]
                buys_idx = valid_indices[buys_relative_idx]
                
                if len(buys_idx) > 0:
                    budget_per_stock = self.cash / len(buys_idx)
                    self.holdings[buys_idx] = budget_per_stock * (1 - self.BUY_COST_RATE)
                    self.cash = 0.0  # 满仓打入
                
        # ================================
        # 3. 盘中涨跌估值变化
        # ================================
        self.holdings = self.holdings * (1.0 + t_returns)
        
        # ================================
        # 4. 统计与奖励分配
        # ================================
        old_asset = self.total_asset
        self.total_asset = self.cash + np.sum(self.holdings)
        
        if self.total_asset > self.peak_asset:
            self.peak_asset = self.total_asset
            
        daily_return = (self.total_asset / old_asset) - 1.0
        reward = float(daily_return * 100.0) 
        
        self.current_step += 1
        done = self.current_step >= len(self.step_indices) - 1
        
        return self._get_state(), reward, done, False, {}

if __name__ == '__main__':
    parquet_dir = "/root/.openclaw/workspace-gemini-assistant/data/daily_parquet"
    macro_csv = "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/macro_states.csv"
    
    train_idx, train_dates, cache = load_numpy_cache(parquet_dir, macro_csv)
    
    print("\nInitializing Vectorized Env V6-1 (1D Sliding Window)...")
    env = DummyVecEnv([lambda: RLTraderEnvV6_1(train_idx, train_dates, cache)])
    
    print("Training PPO Model (V6-1 Slide Window, 80000 steps)...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.01, n_steps=1024, batch_size=64)
    model.learn(total_timesteps=80000)
    
    save_path = "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/adaptive_ppo_v6_1_slide"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
