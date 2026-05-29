import pandas as pd
import numpy as np
import glob, os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

INITIAL_CAPITAL = 100_000_000.0
TARGET_HOLDINGS = 400
SKIP_TOP_PCT = 0.0001
SELL_RANK_PCT = 0.4
BUY_COST_RATE = 0.0008
SELL_COST_RATE = 0.0018

def load_data():
    print("Loading macro states & Rolling Z-Score processing...")
    macro_df = pd.read_csv("/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/macro_states.csv")
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.set_index('date', inplace=True)
    
    macro_cols = [
        'limit_down_rate', 'limit_up_rate', 'up_down_ratio', 'market_breadth',
        'alpha_divergence', 'idx_bias_20', 'idx_vol_ratio_5', 'idx_return'
    ]
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

    print("Building Global 2D Numpy Tensors [days, stocks] to prevent OOM...")
    parquet_dir = "/root/.openclaw/workspace-gemini-assistant/data/daily_parquet"
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    
    valid_dates = [os.path.basename(f).split(".")[0] for f in files if os.path.basename(f).split(".")[0] in macro_dict]
    
    all_stocks = set()
    for f in files:
        d = os.path.basename(f).split(".")[0]
        if d in valid_dates:
            df = pd.read_parquet(f, columns=[])
            all_stocks.update(df.index.tolist())
            
    all_stocks = sorted(list(all_stocks))
    stock2idx = {c: i for i, c in enumerate(all_stocks)}
    date2idx = {d: i for i, d in enumerate(valid_dates)}
    
    N_days = len(valid_dates)
    N_stocks = len(all_stocks)
    
    ret_2d = np.zeros((N_days, N_stocks), dtype=np.float32)
    rank_2d = np.ones((N_days, N_stocks), dtype=np.float32) * 2.0
    is_limit_down_2d = np.zeros((N_days, N_stocks), dtype=bool)
    is_limit_up_2d = np.zeros((N_days, N_stocks), dtype=bool)
    is_halt_2d = np.zeros((N_days, N_stocks), dtype=bool)
    is_st_2d = np.zeros((N_days, N_stocks), dtype=bool)
    valid_mask_2d = np.zeros((N_days, N_stocks), dtype=bool)
    
    for date_str in valid_dates:
        f = os.path.join(parquet_dir, f"{date_str}.parquet")
        df = pd.read_parquet(f)
        d_idx = date2idx[date_str]
        s_indices = [stock2idx[c] for c in df.index]
        
        ret_2d[d_idx, s_indices] = df['raw_return'].fillna(0).values
        rank_2d[d_idx, s_indices] = df['rank_pct'].values
        is_limit_down_2d[d_idx, s_indices] = df['is_limit_down'].values
        is_limit_up_2d[d_idx, s_indices] = df['is_limit_up'].values
        is_halt_2d[d_idx, s_indices] = df['is_halt'].values
        is_st_2d[d_idx, s_indices] = df['is_st'].values
        valid_mask_2d[d_idx, s_indices] = True

    macro_obs_arr = np.zeros((N_days, 8), dtype=np.float32)
    for date_str in valid_dates:
        d_idx = date2idx[date_str]
        m = macro_dict[date_str]
        macro_obs_arr[d_idx] = [
            m['limit_down_rate'], m['limit_up_rate'], m['up_down_ratio'],
            m['market_breadth'], m['alpha_divergence'], m['idx_bias_20'],
            m['idx_vol_ratio_5'], m['idx_return']
        ]

    buy_cand_mask = (
        (rank_2d > SKIP_TOP_PCT) &
        (~is_limit_up_2d) &
        (~is_halt_2d) &
        (~is_st_2d) &
        valid_mask_2d
    )
    
    train_dates = [d for d in valid_dates if d <= "2025-06-30"]
    test_dates = [d for d in valid_dates if d > "2025-06-30"]
    
    env_data = {
        'N_stocks': N_stocks,
        'valid_dates': valid_dates,
        'date2idx': date2idx,
        'macro_obs_arr': macro_obs_arr,
        'ret_2d': ret_2d,
        'rank_2d': rank_2d,
        'is_limit_down_2d': is_limit_down_2d,
        'is_halt_2d': is_halt_2d,
        'valid_mask_2d': valid_mask_2d,
        'buy_cand_mask': buy_cand_mask
    }
    
    return train_dates, test_dates, env_data

class RLTraderEnvV9(gym.Env):
    def __init__(self, date_list, env_data, is_eval=False):
        super(RLTraderEnvV9, self).__init__()
        self.date_list = date_list
        self.env_data = env_data
        self.is_eval = is_eval
        
        # Action: 一维动作空间，同V4，控制预算(Spend Ratio) -> 现金留存
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: 8宏观 + 1回撤水位 + 1现金比例
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
        
    def _get_state(self):
        date_str = self.date_list[self.current_step]
        d_idx = self.env_data['date2idx'][date_str]
        macro_obs = self.env_data['macro_obs_arr'][d_idx]
        
        dd = (self.total_asset - self.peak_asset) / self.peak_asset if self.peak_asset > 0 else 0.0
        dd_z = np.clip(dd * 10.0, -3.0, 3.0) 
        cash_ratio_z = np.clip((self.cash / max(self.total_asset, 1.0)) * 2.0 - 1.0, -3.0, 3.0) 
        
        obs = np.concatenate([macro_obs, [dd_z, cash_ratio_z]]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = INITIAL_CAPITAL
        self.portfolio_values = np.zeros(self.env_data['N_stocks'], dtype=np.float32)
        self.total_asset = INITIAL_CAPITAL
        self.peak_asset = INITIAL_CAPITAL
        self.records = []
        
        # --- DSoR (微分索提诺比率) 在线均值和平方均值初始化 ---
        self.dsor_A = 0.0      # EMA of returns
        self.dsor_B = 1e-4     # EMA of DOWNSIDE squared returns 
        self.target_return = 0.0 # 目标基准收益率 (无风险利率或0)
        self.eta = 0.05        # 衰减平滑系数
        
        return self._get_state(), {}

    def step(self, action):
        # 动作空间和 V4 一模一样：控制花出现金的比例
        act_spend_ratio = np.clip(action[0], 0.0, 1.0)
        
        date_str = self.date_list[self.current_step]
        d_idx = self.env_data['date2idx'][date_str]
        d_env = self.env_data
        
        hold_mask = self.portfolio_values > 0
        
        # 1. Sell
        sell_cond = (
            hold_mask & 
            d_env['valid_mask_2d'][d_idx] & 
            (~d_env['is_limit_down_2d'][d_idx]) & 
            (~d_env['is_halt_2d'][d_idx]) & 
            (d_env['rank_2d'][d_idx] > SELL_RANK_PCT)
        )
        if np.any(sell_cond):
            sell_amount = np.sum(self.portfolio_values[sell_cond])
            self.cash += sell_amount * (1 - SELL_COST_RATE)
            self.portfolio_values[sell_cond] = 0.0
            
        hold_mask = self.portfolio_values > 0
        
        # 2. Buy
        num_held = np.sum(hold_mask)
        num_to_buy = int(TARGET_HOLDINGS - num_held)
        if num_to_buy > 0 and self.cash > 0:
            cand_mask = d_env['buy_cand_mask'][d_idx] & (~hold_mask)
            if np.any(cand_mask):
                cand_ranks = d_env['rank_2d'][d_idx].copy()
                cand_ranks[~cand_mask] = np.inf
                n_cands = np.sum(cand_mask)
                actual_buy_num = min(num_to_buy, n_cands)
                
                if actual_buy_num > 0:
                    buy_indices = np.argpartition(cand_ranks, actual_buy_num - 1)[:actual_buy_num]
                    buy_indices = buy_indices[cand_ranks[buy_indices] != np.inf]
                    if len(buy_indices) > 0:
                        spend_cash = self.cash * act_spend_ratio
                        budget_per_stock = spend_cash / len(buy_indices)
                        self.portfolio_values[buy_indices] += budget_per_stock * (1 - BUY_COST_RATE)
                        self.cash -= spend_cash

        # 3. Value Update
        hold_mask = self.portfolio_values > 0
        update_mask = hold_mask & d_env['valid_mask_2d'][d_idx]
        if np.any(update_mask):
            self.portfolio_values[update_mask] *= (1.0 + d_env['ret_2d'][d_idx][update_mask])
            
        # 4. Stats & DSoR Reward
        old_asset = self.total_asset
        self.total_asset = self.cash + np.sum(self.portfolio_values)
        if self.total_asset > self.peak_asset:
            self.peak_asset = self.total_asset
            
        daily_return = (self.total_asset / old_asset) - 1.0
        
        # --- 核心：计算 DSoR (Differential Sortino Ratio) 奖励 ---
        rt = float(daily_return * 100.0)
        
        # 【与DSR的核心区别】: 只惩罚下行偏离！上涨赚的钱不计入风险方差 B！
        downside = min(0.0, rt - self.target_return)
        
        delta_A = rt - self.dsor_A
        delta_B = downside**2 - self.dsor_B
        
        # DSoR 里的方差项仅由下行偏离组成
        variance = self.dsor_B 
        if variance < 1e-6:
            variance = 1e-6
            
        dsor_reward = (self.dsor_B * delta_A - 0.5 * self.dsor_A * delta_B) / (variance ** 1.5)
        
        # 状态更新
        self.dsor_A += self.eta * delta_A
        self.dsor_B += self.eta * delta_B
        
        # 截断极端反馈
        reward = float(np.clip(dsor_reward, -5.0, 5.0))
        
        if self.is_eval:
            self.records.append({
                'Date': date_str,
                'Total_Capital': self.total_asset,
                'Cum_Return': (self.total_asset / INITIAL_CAPITAL) - 1.0,
                'Action_Spend_Ratio': act_spend_ratio,
                'Cash_Ratio': self.cash / self.total_asset,
                'Holdings_Count': int(np.sum(self.portfolio_values > 0)),
                'DSoR_Reward': dsor_reward
            })
        
        self.current_step += 1
        
        if self.is_eval:
            done = self.current_step >= len(self.date_list)
        else:
            done = self.current_step >= len(self.date_list) - 1
            
        next_state = self._get_state() if self.current_step < len(self.date_list) else np.zeros(10, dtype=np.float32)
        return next_state, reward, done, False, {}

if __name__ == "__main__":
    train_dates, test_dates, env_data = load_data()
    print("Initializing Vectorized Train Environment (V9 - DSoR Reward)...")
    env = DummyVecEnv([lambda: RLTraderEnvV9(train_dates, env_data, is_eval=False)])
    print("Training RL Model...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.01, n_steps=1024, batch_size=64)
    model.learn(total_timesteps=80000)
    model.save("/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/adaptive_ppo_v9_dsor_model")
    print("Training Complete.")
