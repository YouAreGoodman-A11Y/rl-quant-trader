import qlib
from qlib.data import D
import pandas as pd
import numpy as np
import glob, os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 1. Initialize
qlib.init(provider_uri="/data/mamba_qlib_bin", region=qlib.config.REG_CN)

# 2. Parameters
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

# Rolling Z-score (250 days, min_periods=20 to avoid extreme volatility early on)
rolling_mean = macro_df[macro_cols].rolling(window=250, min_periods=20).mean()
rolling_std = macro_df[macro_cols].rolling(window=250, min_periods=20).std()

# Fill early NaNs with expanding window
expanding_mean = macro_df[macro_cols].expanding(min_periods=1).mean()
expanding_std = macro_df[macro_cols].expanding(min_periods=1).std()

rolling_mean = rolling_mean.fillna(expanding_mean).fillna(0)
rolling_std = rolling_std.fillna(expanding_std).fillna(1e-8)
rolling_std[rolling_std == 0] = 1e-8

macro_z = (macro_df[macro_cols] - rolling_mean) / rolling_std
# Clipping to [-3, 3] to prevent gradient explosion from black swan data
macro_z = macro_z.clip(-3.0, 3.0)

macro_z.index = macro_z.index.strftime('%Y-%m-%d')
macro_dict = macro_z.to_dict(orient='index')

print("Loading Features & Predictions...")
predict_dir = "/root/.openclaw/workspace-gemini-assistant/data/predict"
files = sorted(glob.glob(os.path.join(predict_dir, "*.csv")))

instruments = D.instruments(market="all_a_shares") 
fields = [
    "$close", "$adj_factor", "$is_st",
    "Ref($open, -1)", "Ref($high, -1)", "Ref($low, -1)", "Ref($volume, -1)", "Ref($adj_factor, -1)",
    "Ref($up_limit, -1)", "Ref($down_limit, -1)",
    "Ref($open, -2)", "Ref($adj_factor, -2)"
]
df_features = D.features(instruments, fields, start_time="2022-01-01", end_time="2026-03-31", freq="day")
df_features.columns = [
    "T_close", "T_adj", "T_is_st",
    "T1_open", "T1_high", "T1_low", "T1_vol", "T1_adj", 
    "T1_up_limit", "T1_down_limit",
    "T2_open", "T2_adj"
]
df_features = df_features.reset_index()

days_data = []
for file in files:
    date_str = os.path.basename(file).split(".")[0]
    if date_str not in macro_dict: continue
    
    df_pred = pd.read_csv(file)
    def convert_code(c):
        if str(c).endswith('.XSHE'): return 'sz' + str(c)[:6]
        elif str(c).endswith('.XSHG'): return 'sh' + str(c)[:6]
        return c
    df_pred['code_qlib'] = df_pred['code'].apply(convert_code)
    df_pred['rank_pct'] = df_pred['prediction'].rank(pct=True, ascending=False)
    
    df_f = df_features[df_features['datetime'] == date_str]
    if df_f.empty: continue
    
    df = pd.merge(df_pred, df_f, left_on='code_qlib', right_on='instrument', how='inner')
    df.set_index('code_qlib', inplace=True)
    
    df['T_adj'] = df['T_adj'].fillna(1.0)
    df['T1_adj'] = df['T1_adj'].fillna(1.0)
    df['T2_adj'] = df['T2_adj'].fillna(1.0)
    
    df['T_close_adj'] = df['T_close'] * df['T_adj']
    df['T1_open_adj'] = df['T1_open'] * df['T1_adj']
    df['T2_open_adj'] = df['T2_open'] * df['T2_adj']
    
    df['is_limit_up'] = (np.isclose(df['T1_open'], df['T1_high'], rtol=1e-4) & np.isclose(df['T1_open'], df['T1_up_limit'], rtol=1e-4))
    df['is_limit_down'] = (np.isclose(df['T1_open'], df['T1_low'], rtol=1e-4) & np.isclose(df['T1_open'], df['T1_down_limit'], rtol=1e-4))
    df['is_halt'] = (df['T1_vol'].fillna(0) == 0) | (df['T1_open'].isna())
    df['is_st'] = (df['T_is_st'] == 1.0)
    
    df['raw_return'] = (df['T2_open_adj'] / df['T1_open_adj']) - 1
    df['raw_return'] = df['raw_return'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    days_data.append((date_str, df, macro_dict[date_str]))

print(f"Loaded {len(days_data)} days of data.")

class RLTraderEnv(gym.Env):
    def __init__(self, data):
        super(RLTraderEnv, self).__init__()
        self.data = data
        # 动作: [动用可用现金池买新股的比例]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # 状态: 8个盘口宏观指标 + 自身最大回撤水位 + 自身现金仓位 = 10维
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
        
    def _get_state(self):
        _, _, macro = self.data[self.current_step]
        dd = (self.total_asset - self.peak_asset) / self.peak_asset if self.peak_asset > 0 else 0.0
        # 归一化自身状态
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
        return self._get_state(), {}

    def step(self, action):
        # 提取模型输出的连续动作：动用可用现金的百分比
        act_spend_ratio = np.clip(action[0], 0.0, 1.0)
        
        date_str, df, _ = self.data[self.current_step]
        
        # --- 1. 强制卖出阶段 ---
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
            
        # --- 2. 动用资金买入阶段 (应用 RL 动作) ---
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
                # RL Action! 决定动用多少可用资金！
                spend_cash = self.cash * act_spend_ratio
                budget_per_stock = spend_cash / len(buys)
                
                for code in buys:
                    self.portfolio[code] = budget_per_stock * (1 - BUY_COST_RATE)
                    self.cash -= budget_per_stock
                    
        # --- 3. 盘中涨跌估值变化 (T1 -> T2) ---
        for code in list(self.portfolio.keys()):
            if code in df.index:
                self.portfolio[code] *= (1 + df.loc[code, 'raw_return'])
                
        # --- 4. 统计与奖励分配 ---
        old_asset = self.total_asset
        self.total_asset = self.cash + sum(self.portfolio.values())
        if self.total_asset > self.peak_asset:
            self.peak_asset = self.total_asset
            
        daily_return = (self.total_asset / old_asset) - 1.0
        
        # 奖励极简且纯粹：放大了100倍的当日组合绝对净收益率
        reward = float(daily_return * 100.0) 
        
        self.records.append({
            'Date': date_str,
            'Total_Capital': self.total_asset,
            'Cum_Return': (self.total_asset / INITIAL_CAPITAL) - 1.0,
            'Action_Spend_Ratio': act_spend_ratio,
            'Cash_Ratio': self.cash / self.total_asset
        })
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done, False, {}

# 切分训练集与测试集
train_data = [d for d in days_data if d[0] <= "2025-06-30"]
test_data = [d for d in days_data if d[0] > "2025-06-30"]

print("Training RL Model (V1 Action)...")
env = DummyVecEnv([lambda: RLTraderEnv(train_data)])
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.01, n_steps=1024, batch_size=64)
model.learn(total_timesteps=80000)
model.save("/root/.openclaw/workspace-gemini-assistant/rl_trader/adaptive_ppo_v1_action")

print("Evaluating Test Set...")
eval_env = RLTraderEnv(test_data)
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

plt.title(f"RL V1 (1D-Action: Cash Control) Test Set | Final Return: {final_return*100:.2f}%")
plt.tight_layout()
plt.savefig("/root/.openclaw/workspace-gemini-assistant/rl_trader/rl_v1_action_test.png")
print("Done! Plot saved to rl_v1_action_test.png")
