import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gymnasium as gym
from stable_baselines3 import PPO
import os
import glob

# ==========================================
# 1. 重构之前环境类中精简的核心评估逻辑
# ==========================================
class SimpleEvaluator:
    def __init__(self, data_list, macro_dict, ram_cache):
        self.data_list = data_list
        self.macro_dict = macro_dict
        self.ram_cache = ram_cache
        self.INITIAL_CAPITAL = 100_000_000.0
        self.TARGET_HOLDINGS = 400
        self.SKIP_TOP_PCT = 0.0001
        self.SELL_RANK_PCT = 0.4
        self.BUY_COST_RATE = 0.0008
        self.SELL_COST_RATE = 0.0018
        
    def _get_state(self, current_step, total_asset, peak_asset, cash):
        date_str = self.data_list[current_step]
        macro = self.macro_dict[date_str]
        
        dd = (total_asset - peak_asset) / peak_asset if peak_asset > 0 else 0.0
        dd_z = np.clip(dd * 10.0, -3.0, 3.0) 
        cash_ratio_z = np.clip((cash / max(total_asset, 1.0)) * 2.0 - 1.0, -3.0, 3.0) 
        
        obs = np.array([
            macro['limit_down_rate'], macro['limit_up_rate'], macro['up_down_ratio'],
            macro['market_breadth'], macro['alpha_divergence'], macro['idx_bias_20'],
            macro['idx_vol_ratio_5'], macro['idx_return'],
            dd_z, cash_ratio_z
        ], dtype=np.float32)
        return obs

    def evaluate(self, model):
        cash = self.INITIAL_CAPITAL
        portfolio = {}
        total_asset = self.INITIAL_CAPITAL
        peak_asset = self.INITIAL_CAPITAL
        records = []
        
        for current_step in range(len(self.data_list)):
            date_str = self.data_list[current_step]
            obs = self._get_state(current_step, total_asset, peak_asset, cash)
            action, _ = model.predict(obs, deterministic=True)
            act_spend_ratio = np.clip(action[0], 0.0, 1.0)
            
            df = self.ram_cache.get(date_str, None)
            
            if df is not None and not df.empty:
                codes_to_sell = []
                for code in list(portfolio.keys()):
                    if code not in df.index: continue
                    if df.loc[code, 'is_limit_down'] or df.loc[code, 'is_halt']: continue
                    if df.loc[code, 'rank_pct'] > self.SELL_RANK_PCT:
                        codes_to_sell.append(code)
                        
                for code in codes_to_sell:
                    sell_val = portfolio[code]
                    cash += sell_val * (1 - self.SELL_COST_RATE)
                    del portfolio[code]
                    
                num_to_buy = max(0, self.TARGET_HOLDINGS - len(portfolio))
                if num_to_buy > 0 and cash > 0:
                    cand_mask = (
                        (df['rank_pct'] > self.SKIP_TOP_PCT) & 
                        (~df['is_limit_up']) &            
                        (~df['is_halt']) &                
                        (~df['is_st']) &                  
                        (~df.index.isin(portfolio.keys()))
                    )
                    candidates = df[cand_mask].sort_values('rank_pct', ascending=True)
                    buys = candidates.index.tolist()[:num_to_buy]
                    
                    if len(buys) > 0:
                        spend_cash = cash * act_spend_ratio
                        budget_per_stock = spend_cash / len(buys)
                        for code in buys:
                            portfolio[code] = budget_per_stock * (1 - self.BUY_COST_RATE)
                            cash -= budget_per_stock
                            
                for code in list(portfolio.keys()):
                    if code in df.index:
                        portfolio[code] *= (1 + df.loc[code, 'raw_return'])
                    
            total_asset = cash + sum(portfolio.values())
            if total_asset > peak_asset:
                peak_asset = total_asset
                
            records.append({
                'Date': date_str,
                'Total_Capital': total_asset,
                'Cum_Return': (total_asset / self.INITIAL_CAPITAL) - 1.0,
                'Action_Spend_Ratio': act_spend_ratio,
                'Cash_Ratio': cash / total_asset
            })
            
        res_df = pd.DataFrame(records)
        res_df['Date'] = pd.to_datetime(res_df['Date'])
        return res_df

# ==========================================
# 2. 准备数据并加载模型
# ==========================================
print("Loading macro & Parquet data...")
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
ram_cache = {}
valid_dates = []

for f in files:
    date_str = os.path.basename(f).split(".")[0]
    if date_str in macro_dict:
        ram_cache[date_str] = pd.read_parquet(f)
        valid_dates.append(date_str)

train_dates = [d for d in valid_dates if d <= "2025-06-30"]
test_dates = [d for d in valid_dates if d > "2025-06-30"]

print("Loading Model...")
model = PPO.load("/root/.openclaw/workspace-gemini-assistant/rl_trader/adaptive_ppo_v4_ram.zip")

print("Evaluating Train Set...")
train_evaluator = SimpleEvaluator(train_dates, macro_dict, ram_cache)
train_res = train_evaluator.evaluate(model)

print("Evaluating Test Set...")
test_evaluator = SimpleEvaluator(test_dates, macro_dict, ram_cache)
test_res = test_evaluator.evaluate(model)

# ==========================================
# 3. 画图 (遵守风控规范)
# ==========================================
def plot_dataset(res_df, title_prefix, out_path, color):
    res_df['Daily_Return'] = res_df['Total_Capital'].pct_change().fillna(0)
    mean_ret = res_df['Daily_Return'].mean()
    std_ret = res_df['Daily_Return'].std()
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0
    
    rolling_max = res_df['Total_Capital'].cummax()
    drawdown = (res_df['Total_Capital'] / rolling_max) - 1.0
    trough_date = drawdown.idxmin()
    max_dd = drawdown.min()
    peak_date = res_df.loc[:trough_date, 'Total_Capital'].idxmax()
    
    trough_date_val = res_df.loc[trough_date, 'Date']
    peak_date_val = res_df.loc[peak_date, 'Date']
    
    trough_date_str = trough_date_val.strftime('%Y-%m-%d')
    peak_date_str = peak_date_val.strftime('%Y-%m-%d')
    final_return = res_df['Cum_Return'].iloc[-1]
    
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    
    ax1.plot(res_df['Date'], res_df['Cum_Return'] * 100, label=f'RL Strategy Return', color=color, linewidth=2)
    
    ax1.axvspan(peak_date_val, trough_date_val, color='gray', alpha=0.3, label='Max Drawdown Range')
    
    peak_val = res_df.loc[peak_date, 'Cum_Return'] * 100
    trough_val = res_df.loc[trough_date, 'Cum_Return'] * 100
    ax1.scatter(peak_date_val, peak_val, color='green', s=100, zorder=5, marker='v', label=f'Peak')
    ax1.scatter(trough_date_val, trough_val, color='darkred', s=100, zorder=5, marker='^', label=f'Trough')
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    ax2 = ax1.twinx()
    ax2.plot(res_df['Date'], res_df['Action_Spend_Ratio'], label='RL Action: Spend Ratio', color='orange', alpha=0.4, linewidth=1)
    ax2.plot(res_df['Date'], res_df['Cash_Ratio'], label='Real Cash Ratio', color='green', alpha=0.7, linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Ratio [0-1]', color='black')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)
    
    title_text = (f"{title_prefix}\n"
                  f"Cum Return: {final_return*100:.2f}% | "
                  f"Sharpe Ratio: {sharpe_ratio:.2f} | "
                  f"Max Drawdown: {max_dd*100:.2f}% ({peak_date_str} to {trough_date_str})")
    
    plt.xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return', fontsize=12, color=color)
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

plot_dataset(train_res, "RL V4 (Train Set: 2022.01 - 2025.06)", "/root/.openclaw/workspace-gemini-assistant/rl_trader/rl_v4_train_strict.png", "blue")
plot_dataset(test_res, "RL V4 (Test Set: 2025.07 - Present)", "/root/.openclaw/workspace-gemini-assistant/rl_trader/rl_v4_test_strict.png", "red")
