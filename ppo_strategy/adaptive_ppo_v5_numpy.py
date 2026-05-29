import pandas as pd
import numpy as np
import os
import glob
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# =======================================================
# 1. 核心底层：Numpy 向量化缓存系统 (极速加载)
# =======================================================
def load_numpy_cache(parquet_dir, macro_csv, max_stocks=5500):
    print("Loading macro & Parquet data into Numpy Tensors...")
    # Load Macro
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
    # 使用股票代码的映射表，确保每天矩阵同一列对应同一只股票
    stock_to_idx = {}
    
    # 第一遍：收集所有合法的股票代码
    for f in files:
        date_str = os.path.basename(f).split(".")[0]
        if date_str in macro_dict:
            valid_dates.append(date_str)
            df = pd.read_parquet(f, columns=[]) # 仅仅读取 index
            for code in df.index:
                if code not in stock_to_idx:
                    stock_to_idx[code] = len(stock_to_idx)
    
    num_days = len(valid_dates)
    num_stocks = len(stock_to_idx)
    print(f"Total days: {num_days}, Total unique stocks: {num_stocks}")
    
    # 构造 Numpy 张量
    rank_matrix = np.full((num_days, num_stocks), 1.0, dtype=np.float32) # 默认排名倒数(1.0)
    return_matrix = np.zeros((num_days, num_stocks), dtype=np.float32)
    trade_mask = np.zeros((num_days, num_stocks), dtype=bool) # 默认全部不可交易
    sell_mask = np.zeros((num_days, num_stocks), dtype=bool) # 强制不能卖出的(停牌/跌停)
    
    macro_matrix = np.zeros((num_days, 8), dtype=np.float32)

    for i, date_str in enumerate(valid_dates):
        f = os.path.join(parquet_dir, f"{date_str}.parquet")
        df = pd.read_parquet(f)
        
        # 获取这一天出现过的股票在全局里的列索引
        indices = [stock_to_idx[c] for c in df.index]
        
        # 填充矩阵
        rank_matrix[i, indices] = df['rank_pct'].values
        return_matrix[i, indices] = df['raw_return'].values
        
        # 可以买的掩码：没涨停 & 没停牌 & 非ST
        can_buy = (~df['is_limit_up']) & (~df['is_halt']) & (~df['is_st'])
        trade_mask[i, indices] = can_buy.values
        
        # 不能卖的掩码：跌停 或 停牌
        cant_sell = df['is_limit_down'] | df['is_halt']
        sell_mask[i, indices] = cant_sell.values
        
        m = macro_dict[date_str]
        macro_matrix[i] = [
            m['limit_down_rate'], m['limit_up_rate'], m['up_down_ratio'],
            m['market_breadth'], m['alpha_divergence'], m['idx_bias_20'],
            m['idx_vol_ratio_5'], m['idx_return']
        ]
        
    # 切分训练与测试集
    train_idx = [i for i, d in enumerate(valid_dates) if d <= "2025-06-30"]
    test_idx = [i for i, d in enumerate(valid_dates) if d > "2025-06-30"]
    
    train_dates = [valid_dates[i] for i in train_idx]
    
    cache = {
        'rank': rank_matrix,
        'return': return_matrix,
        'trade': trade_mask,
        'sell_lock': sell_mask,
        'macro': macro_matrix,
        'num_stocks': num_stocks
    }
    
    return train_idx, train_dates, cache

# =======================================================
# 2. V5 极速向量化环境 (仅开放 Sell Rank 动作)
# =======================================================
class RLTraderEnvV5(gym.Env):
    def __init__(self, step_indices, dates, cache):
        super(RLTraderEnvV5, self).__init__()
        self.step_indices = step_indices
        self.dates = dates
        self.cache = cache
        
        self.INITIAL_CAPITAL = 100_000_000.0
        self.TARGET_HOLDINGS = 400
        self.SKIP_TOP_PCT = 0.0001
        
        self.BUY_COST_RATE = 0.0008
        self.SELL_COST_RATE = 0.0018
        
        # 动作空间：1维，代表卖出排名的阈值 (Sell_Rank_Pct)，取值范围 [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # 观察空间：8维宏观 + 资金回撤水位 + 自身持有现金水位 = 10维
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
        
    def _get_state(self):
        global_idx = self.step_indices[self.current_step]
        macro_obs = self.cache['macro'][global_idx]
        
        dd = (self.total_asset - self.peak_asset) / self.peak_asset if self.peak_asset > 0 else 0.0
        dd_z = np.clip(dd * 10.0, -3.0, 3.0) 
        cash_ratio_z = np.clip((self.cash / max(self.total_asset, 1.0)) * 2.0 - 1.0, -3.0, 3.0) 
        
        obs = np.append(macro_obs, [dd_z, cash_ratio_z]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.INITIAL_CAPITAL
        # 向量化持仓，长为 5000 的数组
        self.holdings = np.zeros(self.cache['num_stocks'], dtype=np.float32)
        
        self.total_asset = self.INITIAL_CAPITAL
        self.peak_asset = self.INITIAL_CAPITAL
        return self._get_state(), {}

    def step(self, action):
        # AI 输出决定今天的宽严线：只要跌出这根线，无脑强平
        sell_threshold = np.clip(action[0], 0.0, 1.0)
        
        global_idx = self.step_indices[self.current_step]
        
        # 获取当天的 Numpy 数据切片
        t_ranks = self.cache['rank'][global_idx]
        t_returns = self.cache['return'][global_idx]
        t_trade = self.cache['trade'][global_idx]
        t_sell_lock = self.cache['sell_lock'][global_idx]
        
        # ================================
        # 1. 向量化卖出逻辑
        # ================================
        # 当前有持仓的 mask
        has_pos_mask = (self.holdings > 0)
        
        # 需要卖的条件：排名变差 (> sell_threshold) 且没有被锁死 (不是跌停/停牌)
        need_sell_mask = has_pos_mask & (t_ranks > sell_threshold) & (~t_sell_lock)
        
        # 卖出回笼资金
        if np.any(need_sell_mask):
            sold_val = np.sum(self.holdings[need_sell_mask])
            self.cash += sold_val * (1 - self.SELL_COST_RATE)
            self.holdings[need_sell_mask] = 0.0
            has_pos_mask = (self.holdings > 0) # 更新持仓 mask
            
        # ================================
        # 2. 向量化买入逻辑 (永远满仓, 固定前400)
        # ================================
        curr_hold_count = np.sum(has_pos_mask)
        num_to_buy = max(0, int(self.TARGET_HOLDINGS - curr_hold_count))
        
        if num_to_buy > 0 and self.cash > 0:
            # 候选池：没被过滤的票 & 我目前没持仓的票 & 排名大于跳过保护的票
            cand_mask = t_trade & (~has_pos_mask) & (t_ranks > self.SKIP_TOP_PCT)
            
            # 使用 Numpy 排序找到最好的那批人
            # 避免对整个 5000 数组完全排序，先用 where 取出合法的
            valid_indices = np.where(cand_mask)[0]
            if len(valid_indices) > 0:
                valid_ranks = t_ranks[valid_indices]
                # 排序获取相对索引，再取需要的数量
                best_relative_idx = np.argsort(valid_ranks)[:num_to_buy]
                buys_idx = valid_indices[best_relative_idx]
                
                budget_per_stock = self.cash / len(buys_idx)
                # 扣除滑点
                self.holdings[buys_idx] = budget_per_stock * (1 - self.BUY_COST_RATE)
                self.cash -= budget_per_stock * len(buys_idx)
                
        # ================================
        # 3. 盘中涨跌估值变化
        # ================================
        # 持仓市值的变化直接用 Numpy 向量相乘，光速完成
        self.holdings = self.holdings * (1.0 + t_returns)
        
        # ================================
        # 4. 统计与奖励分配
        # ================================
        old_asset = self.total_asset
        self.total_asset = self.cash + np.sum(self.holdings)
        
        if self.total_asset > self.peak_asset:
            self.peak_asset = self.total_asset
            
        daily_return = (self.total_asset / old_asset) - 1.0
        # 极简奖励
        reward = float(daily_return * 100.0) 
        
        self.current_step += 1
        done = self.current_step >= len(self.step_indices) - 1
        
        return self._get_state(), reward, done, False, {}

if __name__ == '__main__':
    parquet_dir = "/root/.openclaw/workspace-gemini-assistant/data/daily_parquet"
    macro_csv = "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/macro_states.csv"
    
    train_idx, train_dates, cache = load_numpy_cache(parquet_dir, macro_csv)
    
    print("\nInitializing Vectorized Env V5...")
    env = DummyVecEnv([lambda: RLTraderEnvV5(train_idx, train_dates, cache)])
    
    print("Training PPO Model (V5 Numpy Vectorized, 80000 steps)...")
    # 为了演示速度，直接 80000 步
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.01, n_steps=1024, batch_size=64)
    model.learn(total_timesteps=80000)
    
    save_path = "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/adaptive_ppo_v5_numpy"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
