import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from stable_baselines3 import PPO
from adaptive_ppo_v9_dsor_train import load_data, RLTraderEnvV9

def plot_v7_standard(records, title, save_path):
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df['Daily_Return'] = df['Total_Capital'].pct_change().fillna(0)
    cum_ret = df['Cum_Return'].iloc[-1]
    
    std_dev = df['Daily_Return'].std()
    sharpe = (df['Daily_Return'].mean() / std_dev) * np.sqrt(252) if std_dev > 0 else 0

    rolling_max = df['Total_Capital'].cummax()
    drawdown = (df['Total_Capital'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    end_date = drawdown.idxmin()
    start_date = df['Total_Capital'].loc[:end_date].idxmax()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    ax1.plot(df.index, df['Cum_Return'], label=f'RL Return (Cum: {cum_ret*100:.2f}%, Sharpe: {sharpe:.2f}, MDD: {max_dd*100:.2f}%)', color='blue', linewidth=1.5)
    ax1.axvspan(start_date, end_date, color='red', alpha=0.15, label='Max Drawdown Range')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    ax1.set_title(title, fontsize=14, fontweight='bold')

    ax2.plot(df.index, df['Action_Spend_Ratio'], label='Action: Spend Ratio', color='orange', alpha=0.8)
    ax2.plot(df.index, df['Cash_Ratio'], label='Actual Cash Ratio', color='black', linewidth=2)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')

    ax3.plot(df.index, df['DSoR_Reward'], label='Step DSoR Reward Signal (Downside Only)', color='green', linewidth=1.0, alpha=0.6)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved evaluation chart to {save_path}")

def run_evaluation(model, env, title_prefix, save_name):
    print(f"Evaluating {title_prefix}...")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    plot_v7_standard(env.records, f"{title_prefix} Performance", save_name)

if __name__ == "__main__":
    model_path = "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/adaptive_ppo_v9_dsor_model.zip"
    train_dates, test_dates, env_data = load_data()
    model = PPO.load(model_path)
    
    train_env = RLTraderEnvV9(train_dates, env_data, is_eval=True)
    run_evaluation(model, train_env, "V9 DSoR Reward (Train Set)", "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/v9_dsor_train_eval.png")

    test_env = RLTraderEnvV9(test_dates, env_data, is_eval=True)
    run_evaluation(model, test_env, "V9 DSoR Reward (Test Set)", "/root/.openclaw/workspace-gemini-assistant/rl_trader/ppo_strategy/v9_dsor_test_eval.png")
    print("Evaluation Complete. V7 standard charts generated.")
