import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from stable_baselines3 import PPO

# 引入训练脚本中的 1D 滑窗逻辑作为评估基础
class RLTraderEnvV6_1_Eval:
    def __init__(self, step_indices, dates, cache):
        self.step_indices = step_indices
        self.dates = dates
        self.cache = cache
        
        self.INITIAL_CAPITAL = 100_000_000.0
        self.SELL_RANK_PCT = 0.4
        self.BUY_COST_RATE = 0.0008
        self.SELL_COST_RATE = 0.0018
        self.WINDOW_WIDTH = 150

    def _get_state(self, current_step, total_asset, peak_asset, cash, holdings):
        global_idx = self.step_indices[current_step]
        macro_obs = self.cache['macro'][global_idx]
        
        dd = (total_asset - peak_asset) / peak_asset if peak_asset > 0 else 0.0
        dd_z = np.clip(dd * 10.0, -3.0, 3.0) 
        
        held_count = np.sum(holdings > 0)
        held_z = np.clip((held_count - 150) / 100.0, -3.0, 3.0)
        
        obs = np.append(macro_obs, [dd_z, held_z]).astype(np.float32)
        return obs

    def evaluate(self, model):
        cash = self.INITIAL_CAPITAL
        holdings = np.zeros(self.cache['num_stocks'], dtype=np.float32)
        total_asset = self.INITIAL_CAPITAL
        peak_asset = self.INITIAL_CAPITAL
        records = []
        
        for current_step in range(len(self.step_indices)):
            date_str = self.dates[current_step]
            global_idx = self.step_indices[current_step]
            
            obs = self._get_state(current_step, total_asset, peak_asset, cash, holdings)
            action, _ = model.predict(obs, deterministic=True)
            
            # Action 映射: 0.0~1.0 -> 0~300 名
            val = np.clip(action[0], 0.0, 1.0)
            start_rank = int(val * 300)
            end_rank = start_rank + self.WINDOW_WIDTH
            
            t_ranks = self.cache['rank'][global_idx]
            t_returns = self.cache['return'][global_idx]
            t_trade = self.cache['trade'][global_idx]
            t_sell_lock = self.cache['sell_lock'][global_idx]
            
            # --- Sell ---
            has_pos_mask = (holdings > 0)
            need_sell_mask = has_pos_mask & (t_ranks > self.SELL_RANK_PCT) & (~t_sell_lock)
            if np.any(need_sell_mask):
                sold_val = np.sum(holdings[need_sell_mask])
                cash += sold_val * (1 - self.SELL_COST_RATE)
                holdings[need_sell_mask] = 0.0
                has_pos_mask = (holdings > 0)
                
            # --- Buy ---
            if cash > 0:
                cand_mask = t_trade & (~has_pos_mask)
                valid_indices = np.where(cand_mask)[0]
                
                if len(valid_indices) > 0:
                    valid_ranks = t_ranks[valid_indices]
                    sorted_relative_idx = np.argsort(valid_ranks)
                    
                    buys_relative_idx = sorted_relative_idx[start_rank:end_rank]
                    buys_idx = valid_indices[buys_relative_idx]
                    
                    if len(buys_idx) > 0:
                        budget_per_stock = cash / len(buys_idx)
                        holdings[buys_idx] = budget_per_stock * (1 - self.BUY_COST_RATE)
                        cash = 0.0
                        
            # --- Intraday Return ---
            holdings = holdings * (1.0 + t_returns)
            
            # --- Stats ---
            total_asset = cash + np.sum(holdings)
            if total_asset > peak_asset:
                peak_asset = total_asset
                
            records.append({
                'Date': date_str,
                'Total_Capital': total_asset,
                'Cum_Return': (total_asset / self.INITIAL_CAPITAL) - 1.0,
                'Start_Rank': start_rank,
                'End_Rank': end_rank,
                'Holdings_Count': np.sum(holdings > 0)
            })
            
        res_df = pd.DataFrame(records)
        res_df['Date'] = pd.to_datetime(res_df['Date'])
        return res_df

# ==========================================
# 3. 遵守刚性纪律画图函数
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
    
    plt.figure(figsize=(16, 9))
    ax1 = plt.gca()
    
    ax1.plot(res_df['Date'], res_df['Cum_Return'] * 100, label=f'RL Return', color=color, linewidth=2)
    ax1.axvspan(peak_date_val, trough_date_val, color='gray', alpha=0.3, label='Max Drawdown Range')
    
    peak_val = res_df.loc[peak_date, 'Cum_Return'] * 100
    trough_val = res_df.loc[trough_date, 'Cum_Return'] * 100
    ax1.scatter(peak_date_val, peak_val, color='green', s=100, zorder=5, marker='v', label=f'Peak')
    ax1.scatter(trough_date_val, trough_val, color='darkred', s=100, zorder=5, marker='^', label=f'Trough')
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    ax2 = ax1.twinx()
    ax2.plot(res_df['Date'], res_df['Start_Rank'], color='purple', alpha=0.5, linewidth=1, label='Buy Start Rank')
    ax2.plot(res_df['Date'], res_df['End_Rank'], color='orange', alpha=0.5, linewidth=1, label='Buy End Rank')
    ax2.fill_between(res_df['Date'], res_df['Start_Rank'], res_df['End_Rank'], color='purple', alpha=0.1, label='Buy Window (150 Width)')
    ax2.set_ylabel('Target Rank Window [0-450]', color='purple')
    
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

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from adaptive_ppo_v6_1_slide import load_numpy_cache
    
    parquet_dir = "/root/.openclaw/workspace-gemini-assistant/data/daily_parquet"
    macro_csv = "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/macro_states.csv"
    
    train_idx, train_dates, cache = load_numpy_cache(parquet_dir, macro_csv)
    
    valid_dates = []
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    for f in files:
        date_str = os.path.basename(f).split(".")[0]
        valid_dates.append(date_str)
        
    test_idx = [i for i, d in enumerate(valid_dates) if d > "2025-06-30"]
    test_dates = [valid_dates[i] for i in test_idx]
    
    print("Loading Model...")
    model_path = "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/adaptive_ppo_v6_1_slide.zip"
    model = PPO.load(model_path)
    
    print("Evaluating Train Set...")
    train_eval = RLTraderEnvV6_1_Eval(train_idx, train_dates, cache)
    train_res = train_eval.evaluate(model)
    
    print("Evaluating Test Set...")
    test_eval = RLTraderEnvV6_1_Eval(test_idx, test_dates, cache)
    test_res = test_eval.evaluate(model)
    
    plot_dataset(train_res, "RL V6-1 (1D Slide Window) Train Set: 2022.01 - 2025.06", "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/rl_v6_1_train_slide.png", "blue")
    plot_dataset(test_res, "RL V6-1 (1D Slide Window) Test Set: 2025.07 - Present", "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/rl_v6_1_test_slide.png", "red")
