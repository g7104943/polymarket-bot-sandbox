@echo off
REM 批量回测8套模型 (Windows 版本)

REM 创建报告目录
if not exist "data\reports" mkdir "data\reports"

REM 回测参数
set INITIAL_CAPITAL=400
set BET_RATIO=0.05
set PROB_THRESHOLD=0.92
set TEST_DAYS=90

REM 风险控制参数（避免大回撤）
set MAX_DRAWDOWN_LIMIT=0.5
set STOP_LOSS_PCT=0.3
set CONSECUTIVE_LOSS_LIMIT=5
set CAPITAL_PROTECTION_PCT=0.5

echo ============================================================
echo 批量回测8套模型
echo ============================================================
echo 初始资金: $%INITIAL_CAPITAL%
echo 下注比例: 5%%
echo 置信度阈值: 92%%
echo 回测天数: %TEST_DAYS% 天
echo 风险控制: 最大回撤50%%, 止损30%%, 连续亏损5次, 资金保护50%%
echo ============================================================
echo.

set SUCCESS=0
set FAILED=0

REM 1. models (旧特征集)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 回测: 1. models (旧特征集)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/backtest_simulation.py --initial-capital %INITIAL_CAPITAL% --bet-ratio %BET_RATIO% --prob-threshold %PROB_THRESHOLD% --test-days %TEST_DAYS% --models-dir data/models --max-drawdown-limit %MAX_DRAWDOWN_LIMIT% --stop-loss-pct %STOP_LOSS_PCT% --consecutive-loss-limit %CONSECUTIVE_LOSS_LIMIT% --capital-protection-pct %CAPITAL_PROTECTION_PCT% --output-html data/reports/backtest_models.html
if %ERRORLEVEL% EQU 0 (set /a SUCCESS+=1) else (set /a FAILED+=1)
echo.

REM 2. models_A (旧特征集)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 回测: 2. models_A (旧特征集)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/backtest_simulation.py --initial-capital %INITIAL_CAPITAL% --bet-ratio %BET_RATIO% --prob-threshold %PROB_THRESHOLD% --test-days %TEST_DAYS% --models-dir data/models_A --max-drawdown-limit %MAX_DRAWDOWN_LIMIT% --stop-loss-pct %STOP_LOSS_PCT% --consecutive-loss-limit %CONSECUTIVE_LOSS_LIMIT% --capital-protection-pct %CAPITAL_PROTECTION_PCT% --output-html data/reports/backtest_models_A.html
if %ERRORLEVEL% EQU 0 (set /a SUCCESS+=1) else (set /a FAILED+=1)
echo.

REM 3. models_B (新特征集 v4)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 回测: 3. models_B (新特征集 v4)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/backtest_simulation.py --initial-capital %INITIAL_CAPITAL% --bet-ratio %BET_RATIO% --prob-threshold %PROB_THRESHOLD% --test-days %TEST_DAYS% --models-dir data/models_B --max-drawdown-limit %MAX_DRAWDOWN_LIMIT% --stop-loss-pct %STOP_LOSS_PCT% --consecutive-loss-limit %CONSECUTIVE_LOSS_LIMIT% --capital-protection-pct %CAPITAL_PROTECTION_PCT% --output-html data/reports/backtest_models_B.html
if %ERRORLEVEL% EQU 0 (set /a SUCCESS+=1) else (set /a FAILED+=1)
echo.

REM 4. models_C (新特征集 v4)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 回测: 4. models_C (新特征集 v4)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/backtest_simulation.py --initial-capital %INITIAL_CAPITAL% --bet-ratio %BET_RATIO% --prob-threshold %PROB_THRESHOLD% --test-days %TEST_DAYS% --models-dir data/models_C --max-drawdown-limit %MAX_DRAWDOWN_LIMIT% --stop-loss-pct %STOP_LOSS_PCT% --consecutive-loss-limit %CONSECUTIVE_LOSS_LIMIT% --capital-protection-pct %CAPITAL_PROTECTION_PCT% --output-html data/reports/backtest_models_C.html
if %ERRORLEVEL% EQU 0 (set /a SUCCESS+=1) else (set /a FAILED+=1)
echo.

REM 5. models_v4 (新特征集 v4)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 回测: 5. models (新特征集 v4)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/backtest_simulation.py --initial-capital %INITIAL_CAPITAL% --bet-ratio %BET_RATIO% --prob-threshold %PROB_THRESHOLD% --test-days %TEST_DAYS% --models-dir data/models_v4 --max-drawdown-limit %MAX_DRAWDOWN_LIMIT% --stop-loss-pct %STOP_LOSS_PCT% --consecutive-loss-limit %CONSECUTIVE_LOSS_LIMIT% --capital-protection-pct %CAPITAL_PROTECTION_PCT% --output-html data/reports/backtest_models_v4.html
if %ERRORLEVEL% EQU 0 (set /a SUCCESS+=1) else (set /a FAILED+=1)
echo.

REM 6. models_A_v4 (新特征集 v4)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 回测: 6. models_A (新特征集 v4)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/backtest_simulation.py --initial-capital %INITIAL_CAPITAL% --bet-ratio %BET_RATIO% --prob-threshold %PROB_THRESHOLD% --test-days %TEST_DAYS% --models-dir data/models_A_v4 --max-drawdown-limit %MAX_DRAWDOWN_LIMIT% --stop-loss-pct %STOP_LOSS_PCT% --consecutive-loss-limit %CONSECUTIVE_LOSS_LIMIT% --capital-protection-pct %CAPITAL_PROTECTION_PCT% --output-html data/reports/backtest_models_A_v4.html
if %ERRORLEVEL% EQU 0 (set /a SUCCESS+=1) else (set /a FAILED+=1)
echo.

REM 7. models_B_old (旧特征集)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 回测: 7. models_B (旧特征集)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/backtest_simulation.py --initial-capital %INITIAL_CAPITAL% --bet-ratio %BET_RATIO% --prob-threshold %PROB_THRESHOLD% --test-days %TEST_DAYS% --models-dir data/models_B_old --max-drawdown-limit %MAX_DRAWDOWN_LIMIT% --stop-loss-pct %STOP_LOSS_PCT% --consecutive-loss-limit %CONSECUTIVE_LOSS_LIMIT% --capital-protection-pct %CAPITAL_PROTECTION_PCT% --output-html data/reports/backtest_models_B_old.html
if %ERRORLEVEL% EQU 0 (set /a SUCCESS+=1) else (set /a FAILED+=1)
echo.

REM 8. models_C_old (旧特征集)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 回测: 8. models_C (旧特征集)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/backtest_simulation.py --initial-capital %INITIAL_CAPITAL% --bet-ratio %BET_RATIO% --prob-threshold %PROB_THRESHOLD% --test-days %TEST_DAYS% --models-dir data/models_C_old --max-drawdown-limit %MAX_DRAWDOWN_LIMIT% --stop-loss-pct %STOP_LOSS_PCT% --consecutive-loss-limit %CONSECUTIVE_LOSS_LIMIT% --capital-protection-pct %CAPITAL_PROTECTION_PCT% --output-html data/reports/backtest_models_C_old.html
if %ERRORLEVEL% EQU 0 (set /a SUCCESS+=1) else (set /a FAILED+=1)
echo.

REM 汇总
echo ============================================================
echo 回测完成汇总
echo ============================================================
echo 成功: %SUCCESS% 套
echo 失败: %FAILED% 套
echo 报告保存在: data\reports\
echo ============================================================
pause
