import qlib
from qlib.data import D
import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ==========================================
# 1. 初始化 Qlib 数据引擎
# ==========================================
qlib.init(provider_uri="/data/mamba_qlib_bin", region=qlib.config.REG_CN)

# ==========================================
# 2. 设置路径与核心参数
# ==========================================
predict_dir = "/root/.openclaw/workspace-gemini-assistant/data/predict"
files = sorted(glob.glob(os.path.join(predict_dir, "*.csv")))

INITIAL_CAPITAL = 100_000_000.0  # 初始资金 1 亿
TARGET_HOLDINGS = 400            # 目标持有 400 只股票
SKIP_TOP_PCT = 0.01            # 不买排名前百分之0.01
SELL_RANK_PCT = 0.4              # 卖出排名在 0.4 之外的股票

# 交易成本参数 (实际费率)
BUY_FEE = 0.0003
SELL_FEE = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.0005

BUY_COST_RATE = BUY_FEE + SLIPPAGE                 # 买入综合费率 = 0.0008
SELL_COST_RATE = SELL_FEE + STAMP_TAX + SLIPPAGE   # 卖出综合费率 = 0.0018

# ==========================================
# 3. 加载行情特征并拼接预测数据
# ==========================================
print("Loading features from Qlib...")
instruments = D.instruments(market="all")
# T_close: T0 收盘价 (用于涨跌停判定基准)
# T1_open: T1 开盘价 (用于实际交易买卖的价格)
# T2_open: T2 开盘价 (用于 T1->T2 持仓过夜的估值更新)
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
print("Merging predictions with features...")
for file in files:
    date_str = os.path.basename(file).split(".")[0]
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
    
    # 填补可能缺失的复权因子，避免乘法产生 NaN
    df['T_adj'] = df['T_adj'].fillna(1.0)
    df['T1_adj'] = df['T1_adj'].fillna(1.0)
    df['T2_adj'] = df['T2_adj'].fillna(1.0)
    
    # 计算复权价
    df['T_close_adj'] = df['T_close'] * df['T_adj']
    df['T1_open_adj'] = df['T1_open'] * df['T1_adj']
    df['T2_open_adj'] = df['T2_open'] * df['T2_adj']
    
    # 判定涨跌停与停牌 (严格使用 Qlib 原生的每日涨跌停价格)
    # is_limit_up: T1 日一字涨停 (开盘价 == 最高价 == 涨停价)
    df['is_limit_up'] = (np.isclose(df['T1_open'], df['T1_high'], rtol=1e-4) & np.isclose(df['T1_open'], df['T1_up_limit'], rtol=1e-4))
    # is_limit_down: T1 日一字跌停 (开盘价 == 最低价 == 跌停价)
    df['is_limit_down'] = (np.isclose(df['T1_open'], df['T1_low'], rtol=1e-4) & np.isclose(df['T1_open'], df['T1_down_limit'], rtol=1e-4))
    df['is_halt'] = (df['T1_vol'].fillna(0) == 0) | (df['T1_open'].isna())
    
    # 真实 ST 判定 (基于 T0 日的 ST 状态)
    df['is_st'] = (df['T_is_st'] == 1.0)
    
    # 单日回报率：基于复权价格，精准反映分红送转
    df['raw_return'] = (df['T2_open_adj'] / df['T1_open_adj']) - 1
    df['raw_return'] = df['raw_return'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    days_data.append((date_str, df))

# ==========================================
# 4. 真实资金流转回测 (Real Cash Ledger)
# ==========================================
print("Starting real cash ledger simulation...")
cash = INITIAL_CAPITAL
# portfolio 记录单票持仓价值。这里为了简化“一手”的碎股运算，我们追踪单只股票的【名义持有金额】
# 卖出时完全清空该票金额，买入时注入金额。自然日间金额随 raw_return 涨跌。
portfolio = {} 
records = []
peak_total_asset = INITIAL_CAPITAL

for i in range(len(days_data)):
    date_str, df = days_data[i]
    
    # -----------------------
    # Step A: 执行卖出操作 (清理退市、掉队票)
    # -----------------------
    sell_count = 0
    codes_to_sell = []
    
    for code in list(portfolio.keys()):
        if code not in df.index:
            # 数据缺失或退市，暂时无法卖出或按估值0卖出（这里作被动保留，假定资金暂时冻结）
            continue
            
        is_ld = df.loc[code, 'is_limit_down']
        is_halt = df.loc[code, 'is_halt']
        rank_pct = df.loc[code, 'rank_pct']
        
        # 强制不卖：一字跌停或停牌
        if is_ld or is_halt:
            continue
            
        # 主动卖出：排名跌出 40%
        if rank_pct > SELL_RANK_PCT:
            codes_to_sell.append(code)
            
    # 执行卖出结算
    for code in codes_to_sell:
        # 该笔股票在 T1 开盘价卖出，其价值即为昨晚结算传过来的金额
        sell_amount = portfolio[code] 
        # 扣除卖出成本
        net_cash_received = sell_amount * (1 - SELL_COST_RATE)
        cash += net_cash_received
        del portfolio[code]
        sell_count += 1
        
    # -----------------------
    # Step B: 执行买入操作 (用可用现金补齐 400 只)
    # -----------------------
    num_to_buy = max(0, TARGET_HOLDINGS - len(portfolio))
    buy_count = 0
    
    if num_to_buy > 0 and cash > 0:
        cand_mask = (
            (df['rank_pct'] > SKIP_TOP_PCT) & 
            (~df['is_limit_up']) &            
            (~df['is_halt']) &                
            (~df['is_st']) &                  
            (~df.index.isin(portfolio.keys()))
        )
        candidates = df[cand_mask].sort_values('rank_pct', ascending=True)
        buys = candidates.index.tolist()[:num_to_buy]
        
        if len(buys) > 0:
            # 可用现金平分给新买的票
            # 真实情况下，为了防爆仓会留一点点buffer，这里我们允许全资金分配
            budget_per_stock = cash / len(buys)
            
            for code in buys:
                # 扣除买入成本，计算实际变成股票仓位的净额
                invested_amount = budget_per_stock * (1 - BUY_COST_RATE)
                portfolio[code] = invested_amount
                cash -= budget_per_stock
                buy_count += 1

    # -----------------------
    # Step C: 盘中涨跌 (T1 -> T2 开盘) 估值更新
    # -----------------------
    for code in list(portfolio.keys()):
        if code in df.index:
            daily_ret = df.loc[code, 'raw_return']
            portfolio[code] *= (1 + daily_ret)
            
    # 计算 T2 开盘时点的总资产
    stock_value = sum(portfolio.values())
    total_asset = cash + stock_value
    
    # -----------------------
    # Step D: 记录当日状态
    # -----------------------
    if total_asset > peak_total_asset:
        peak_total_asset = total_asset
        
    drawdown = (total_asset - peak_total_asset) / peak_total_asset
    cum_return = (total_asset / INITIAL_CAPITAL) - 1.0
    
    # 因为是真实资金累计，当日收益率倒推即可
    prev_asset = records[-1]['Total_Capital'] if len(records) > 0 else INITIAL_CAPITAL
    daily_change_rate = (total_asset / prev_asset) - 1.0
    
    # 换手率：(卖出数量 / 昨天持有总数) 或 (买入数量 / 昨天持有总数)，理解为 Turnover_Rate = 买的股票数 / 当日总持仓股票数。此处统一用 Buy_Count / 400 计算标准换手。
    turnover_rate = buy_count / max(1, len(portfolio))
    
    records.append({
        'Date': date_str,
        'Total_Capital': total_asset,
        'Cash_Balance': cash,
        'Stock_Value': stock_value,
        'Daily_Change_Rate': daily_change_rate,
        'Holdings_Count': len(portfolio),
        'Sell_Count': sell_count,
        'Buy_Count': buy_count,
        'Cum_Return': cum_return,
        'Drawdown': drawdown,
        'Turnover_Rate': turnover_rate
    })

# ==========================================
# 5. 输出指标与图表
# ==========================================
print("Calculating metrics and generating plot...")
df_results = pd.DataFrame(records)

final_return = df_results['Cum_Return'].iloc[-1]
max_drawdown = df_results['Drawdown'].min()
daily_returns_arr = df_results['Daily_Change_Rate'].values
sharpe_ratio = (np.mean(daily_returns_arr) / np.std(daily_returns_arr)) * np.sqrt(252) if np.std(daily_returns_arr) != 0 else 0

mdd_trough_idx = df_results['Drawdown'].idxmin()
mdd_trough_date = df_results.loc[mdd_trough_idx, 'Date']
peak_idx = df_results.loc[:mdd_trough_idx, 'Total_Capital'].idxmax()
mdd_peak_date = df_results.loc[peak_idx, 'Date']

print(f"=====================================")
print(f"累计收益率 (Cumulative Return): {final_return * 100:.2f}%")
print(f"最大回撤 (Max Drawdown): {max_drawdown * 100:.2f}%")
print(f"年化夏普比率 (Sharpe Ratio): {sharpe_ratio:.2f}")
print(f"回撤区间: {mdd_peak_date} -> {mdd_trough_date}")
print(f"=====================================")

output_excel = "/root/.openclaw/workspace-gemini-assistant/rl_trader/strategy_v3_realcash_log.xlsx"
try:
    df_results.to_excel(output_excel, index=False)
    print(f"Results saved to Excel: {output_excel}")
except ImportError:
    fallback_csv = output_excel.replace('.xlsx', '.csv')
    df_results.to_csv(fallback_csv, index=False)

df_results['Date'] = pd.to_datetime(df_results['Date'])

# 双 Y 轴画图：累计收益率 + 换手率
fig, ax1 = plt.subplots(figsize=(14, 7))

# 绘制左轴 (累计收益率)
ax1.plot(df_results['Date'], df_results['Cum_Return'] * 100, label=f'Cumulative Return', color='#1f77b4', linewidth=2)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Cumulative Return (%)', fontsize=12, color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(100))
ax1.grid(True, linestyle='--', alpha=0.7)

# 在图上标注最大回撤区间
peak_dt = pd.to_datetime(mdd_peak_date)
trough_dt = pd.to_datetime(mdd_trough_date)
ax1.axvspan(peak_dt, trough_dt, color='gray', alpha=0.3, label=f'Max DD Period ({max_drawdown*100:.2f}%)')

# 绘制右轴 (换手率)
ax2 = ax1.twinx()
# 使用 14天滑动平均平滑换手率，避免每天剧烈震荡导致看不清趋势
smoothed_turnover = df_results['Turnover_Rate'].rolling(window=14).mean() * 100
ax2.plot(df_results['Date'], smoothed_turnover, label='Daily Turnover Rate (14D MA)', color='#ff7f0e', linewidth=1.5, alpha=0.8)
ax2.set_ylabel('Turnover Rate (%)', fontsize=12, color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(100))

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

plt.title(f'Strategy V3 Return & Turnover\nSharpe Ratio: {sharpe_ratio:.2f} | Final Return: {final_return*100:.2f}%', fontsize=14)
plt.tight_layout()

output_plot = "/root/.openclaw/workspace-gemini-assistant/rl_trader/strategy_v3_realcash_curve.png"
plt.savefig(output_plot, dpi=150)
print(f"Plot saved to {output_plot}")
print("Done.")