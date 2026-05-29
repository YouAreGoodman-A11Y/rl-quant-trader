import pandas as pd
import numpy as np
import glob, os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Parameters
INITIAL_CAPITAL = 100_000_000.0
TARGET_HOLDINGS = 400
SKIP_TOP_PCT = 0.0001
SELL_RANK_PCT = 0.4
BUY_COST_RATE = 0.0008
SELL_COST_RATE = 0.0018

print("Loading macro states & Rolling Z-Score processing...")
macro_df = pd.read_csv("/root/.openclaw/workspace-gemini-assistant/rl_trader/macro_states.csv")
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

parquet_dir = "/root/.openclaw/workspace-gemini-assistant/data/daily_parquet"
files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
valid_dates = [os.path.basename(f).split(".")[0] for f in files if os.path.basename(f).split(".")[0] in macro_dict]

train_dates = [d for d in valid_dates if d <= "2025-06-30"]
test_dates = [d for d in valid_dates if d > "2025-06-30"]

class RLTraderEnvV3(gym.Env):
    def __init__(self, date_list, macro_dict, parquet_dir):
        super(RLTraderEnvV3, self).__init__()
        self.date_list = date_list
        self.macro_dict = macro_dict
        self.parquet_dir = parquet_dir
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
        self.today_df = None
        
    def _load_today_data(self):
        date_str = self.date_list[self.current_step]
        file_path = os.path.join(self.parquet_dir, f"{date_str}.parquet")
        if os.path.exists(file_path):
            self.today_df = pd.read_parquet(file_path)
        else:
            self.today_df = pd.DataFrame()

    def _get_state(self):
        date_str = self.date_list[self.current_step]
        macro = self.macro_dict[date_str]
        
        dd = (self.total_asset - self.peak_asset) / self.peak_asset if self.peak_asset > 0 else 0.0
        dd_z = np.clip(dd * 10.0, -3.0, 3.0) 
        cash_ratio_z = np.clip((self.cash / max(self.total_asset, 1.0)) * 2.0 - 1.0, -3.0, 3.0) 
        
        obs = np.array([
            macro['limit_down_rate'], macro['limit_up_rate'], macro['up_down_ratio'],
            macro['market_breadth'], macro['alpha_divergence'], macro['idx_bias_20'],
            macro['idx_vol_ratio_5'], macro['idx_return'],
            dd_z, cash_ratio_z
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = INITIAL_CAPITAL
        self.portfolio = {}
        self.total_asset = INITIAL_CAPITAL
        self.peak_asset = INITIAL_CAPITAL
        self.records = []
        self._load_today_data()
        return self._get_state(), {}

    def step(self, action):
        act_spend_ratio = np.clip(action[0], 0.0, 1.0)
        date_str = self.date_list[self.current_step]
        df = self.today_df
        
        if df is not None and not df.empty:
            # --- 1. Sell ---
            codes_to_sell = []
            for code in list(self.portfolio.keys()):
                if code not in df.index: continue
                if df.loc[code, 'is_limit_down'] or df.loc[code, 'is_halt']: continue
                if df.loc[code, 'rank_pct'] > SELL_RANK_PCT:
                    codes_to_sell.append(code)
                    
            for code in codes_to_sell:
                sell_val = self.portfolio[code]
                self.cash += sell_val * (1 - SELL_COST_RATE)
                del self.portfolio[code]
                
            # --- 2. Buy ---
            num_to_buy = max(0, TARGET_HOLDINGS - len(self.portfolio))
            if num_to_buy > 0 and self.cash > 0:
                cand_mask = (
                    (df['rank_pct'] > SKIP_TOP_PCT) & 
                    (~df['is_limit_up']) &            
                    (~df['is_halt']) &                
                    (~df['is_st']) &                  
                    (~df.index.isin(self.portfolio.keys()))
                )
                candidates = df[cand_mask].sort_values('rank_pct', ascending=True)
                buys = candidates.index.tolist()[:num_to_buy]
                
                if len(buys) > 0:
                    spend_cash = self.cash * act_spend_ratio
                    budget_per_stock = spend_cash / len(buys)
                    for code in buys:
                        self.portfolio[code] = budget_per_stock * (1 - BUY_COST_RATE)
                        self.cash -= budget_per_stock
                        
            # --- 3. Intraday Value ---
            for code in list(self.portfolio.keys()):
                if code in df.index:
                    self.portfolio[code] *= (1 + df.loc[code, 'raw_return'])
                
        # --- 4. Stats ---
        old_asset = self.total_asset
        self.total_asset = self.cash + sum(self.portfolio.values())
        if self.total_asset > self.peak_asset:
            self.peak_asset = self.total_asset
            
        daily_return = (self.total_asset / old_asset) - 1.0
        reward = float(daily_return * 100.0) 
        
        self.records.append({
            'Date': date_str,
            'Total_Capital': self.total_asset,
            'Cum_Return': (self.total_asset / INITIAL_CAPITAL) - 1.0,
            'Action_Spend_Ratio': act_spend_ratio,
            'Cash_Ratio': self.cash / self.total_asset
        })
        
        self.current_step += 1
        done = self.current_step >= len(self.date_list) - 1
        if not done:
            self._load_today_data()
        
        return self._get_state(), reward, done, False, {}

print("Training RL Model (V3 Parquet Cached)...")
env = DummyVecEnv([lambda: RLTraderEnvV3(train_dates, macro_dict, parquet_dir)])
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.01, n_steps=1024, batch_size=64)
model.learn(total_timesteps=80000)
model.save("/root/.openclaw/workspace-gemini-assistant/rl_trader/adaptive_ppo_v3_parquet")

print("Evaluating Test Set...")
eval_env = RLTraderEnvV3(test_dates, macro_dict, parquet_dir)
obs, _ = eval_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = eval_env.step(action)
    
res_df = pd.DataFrame(eval_env.records)
res_df['Date'] = pd.to_datetime(res_df['Date'])
final_return = res_df['Cum_Return'].iloc[-1]

fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(res_df['Date'], res_df['Cum_Return']*100, label='RL Test Return', color='red', linewidth=2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Return (%)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.plot(res_df['Date'], res_df['Action_Spend_Ratio'], label='Action: Spend Ratio', color='blue', alpha=0.3, linewidth=1)
ax2.plot(res_df['Date'], res_df['Cash_Ratio'], label='Account Cash Ratio', color='green', alpha=0.6, linestyle='--', linewidth=1.5)
ax2.set_ylabel('Ratio [0-1]', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title(f"RL V3 (Parquet Cached) Test Set | Final Return: {final_return*100:.2f}%")
plt.tight_layout()
plt.savefig("/root/.openclaw/workspace-gemini-assistant/rl_trader/rl_v3_parquet_test.png")
print("Done! Plot saved to rl_v3_parquet_test.png")