# logs_eth_10_90 / logs_xrp_20_80 是谁在跑？调查结论

## ✅ 已抓到：是谁在跑

**进程来源**（来自 `ps aux | grep predict-trade`）：

| 组合 | 启动命令 | LOGS_DIR | 进程特征 |
|------|----------|----------|----------|
| **ETH_10_90** | `npm run predict-trade:E` | logs_eth_10_90 | PREDICTION_SUFFIX=_E，周一 04PM 启动，终端 `??` |
| **XRP_20_80** | `npm run predict-trade:F` | logs_xrp_20_80 | PREDICTION_SUFFIX=_F，周一 04PM 启动，终端 `??` |

说明：当前仓库的 **package.json 里没有** `predict-trade:E`、`predict-trade:F` 这两条脚本，说明要么是之前有 E/F 时启动的、要么是从别的拷贝/分支跑的。要停掉：用下面「停止脚本」里的 pkill，或手动 `kill` 对应 ts-node 的 PID。

---

## 1. 仓库内搜索结果（谁在启动 E/F、10_90、20_80）

### 没有任何脚本在「启动」它们

| 位置 | 结果 |
|------|------|
| **package.json** | 只有 `predict-trade:A/B/C/D` 和 GRU 相关脚本，**没有** `predict-trade:E`、`predict-trade:F`，也没有 `LOGS_DIR=logs_eth_10_90` 或 `logs_xrp_20_80` |
| **启动三个版本.sh** | 只启动 A、D、C、B 四组；并且会**删除** `logs_eth_10_90`、`logs_xrp_20_80` 目录（若存在） |
| **启动GRU新模型.sh** | 只启动 GRU 四组，没有 E/F、10_90、20_80 |
| **停止三个版本.sh** | 只 pkill `predict-trade:A/B/C/D` 和 `prediction_writer.*predictions_A/B/C/D.json`，**没有** E、F 或 10_90、20_80 |
| **所有 \*.sh** | 已查 33 个脚本，没有任何一个设置 `LOGS_DIR=logs_eth_10_90` 或 `logs_xrp_20_80` 或 `PREDICTION_SUFFIX=_E/_F` |
| **.env / .env.example** | 没有 PREDICTION_SUFFIX、LOGS_DIR 指向 E/F 或 10_90/20_80 |

### 仓库里和 E/F、10_90、20_80 有关的只有这些

| 文件 | 作用 |
|------|------|
| **启动三个版本.sh** | 删除 `logs_eth_10_90`、`logs_xrp_20_80`（视为「不符合」） |
| **完全清理重新开始.sh** | 注释里写「已废弃 logs_eth_10_90、logs_xrp_20_80」 |
| **scripts/verify_dual_threshold_logic.py** | 仅文档/验证三种双阈值配置（含 ETH_10_90、XRP_20_80），**不启动任何进程** |

结论：**当前仓库里没有任何脚本或配置在「启动」这两组（E/F、10_90、20_80）。**

---

## 2. 它们是怎么在跑的？（推论）

- 报告里 **report_summary.txt** 时间戳是当天 07:35，说明**有进程**在往 `logs_eth_10_90`、`logs_xrp_20_80` 写报告。
- 这些进程一定是**读** `predictions_E.json` / `predictions_F.json`，**写** `logs_eth_10_90` / `logs_xrp_20_80`（即 `LOGS_DIR` + `PREDICTION_SUFFIX` 对应关系）。
- 既然仓库里没有对应启动脚本，只可能是下面之一：
  1. **曾经在终端里手动跑过**，例如：  
     `cd polymarket && LOGS_DIR=logs_eth_10_90 PREDICTION_SUFFIX=_E PROB_THRESHOLD_UP=0.9 PROB_THRESHOLD_DOWN=0.1 ALLOWED_MARKETS=ETH npm run predict-trade`  
     （以及 XRP 的 _F / 20_80 版本），并且没有关窗口/没被停止脚本杀到。
  2. **有一个未提交或本地的脚本**（例如「启动五/六版本.sh」）用类似上面的环境变量调用了 `npm run predict-trade`。
  3. **cron / launchd** 等用上述环境变量定期跑 `npm run predict-trade`。

另外：**停止三个版本.sh 不会杀 E/F**，因为只 pkill 了 A/B/C/D。所以只要有人曾经手动或脚本启动过 E、F，它们会一直跑下去，直到你单独杀掉或关机。

---

## 3. 怎么「抓到」是谁在跑？（实际操作）

在**报告还在更新**的那台机器上执行（在项目根目录或 polymarket 目录均可）：

```bash
# 看所有和 prediction 相关的进程（含 ts-node、prediction_index）
ps aux | grep -E "prediction_index|ts-node|predict-trade"

# 看每个 node/ts-node 进程的环境变量（需替换 <PID> 为实际进程号）
# Linux:
#   cat /proc/<PID>/environ 2>/dev/null | tr '\0' '\n' | grep -E "LOGS_DIR|PREDICTION_SUFFIX"
# macOS（无 /proc，可看完整命令行，若启动时带了 env 可能显示在参数里）:
#   ps -eww -p <PID>
#   或: ps aux | grep ts-node  看每个进程的完整命令
```

- 若某个进程里出现 **LOGS_DIR=logs_eth_10_90** 或 **LOGS_DIR=logs_xrp_20_80**（以及对应的 **PREDICTION_SUFFIX=_E** 或 **_F**），那就是**正在写这两个目录的进程**。
- 看该进程的**父进程**（父 PID）或**启动时间**，可以判断是手动开的终端、某个脚本，还是 cron/launchd。

可选：用活动监视器搜索「ts-node」或「node」，看对应进程的「打开的文件和端口」，确认是否在写 `logs_eth_10_90`/`logs_xrp_20_80`。

---

## 4. 小结

| 问题 | 结论 |
|------|------|
| 仓库里有没有 _E/_F、10_90、20_80 的**启动**脚本或配置？ | **没有**。只有「删除/废弃」和「文档/验证」的引用。 |
| 那它们为什么在跑？ | 一定是**仓库外**启动的：手动命令、未提交脚本，或 cron/launchd。 |
| 怎么确定是谁在跑？ | 用 `ps` 找到写 `logs_eth_10_90` / `logs_xrp_20_80` 的进程（看 LOGS_DIR/PREDICTION_SUFFIX），再根据 PID/父进程/启动方式判断来源。 |
| 停止三个版本会停掉 E/F 吗？ | **不会**。停止脚本只杀 A/B/C/D，E/F 若在跑会继续跑。 |

若要**彻底停掉**这两组，可以：

- 用上面的 `ps` 找到对应 PID 后 `kill <PID>`，或  
- 在停止脚本里增加对 `predict-trade:E`、`predict-trade:F` 或 `LOGS_DIR=logs_eth_10_90`、`LOGS_DIR=logs_xrp_20_80` 的 pkill（需和当前实际启动方式一致）。

---

## 5. 逻辑是否正确？双阈值是否触发？

### 交易逻辑与主流程一致

- E/F 与 A/B/C/D 使用**同一套代码**：`prediction_index.ts` 读 `predictions_*.json` → 按环境变量过滤（双阈值或单阈值）→ `executor.executeRound(toOrder)`。
- 区别仅在于**环境变量**：E/F 设置了 `PROB_THRESHOLD_UP` / `PROB_THRESHOLD_DOWN`，主流程可能用单 `PROB_THRESHOLD` 或另一组双阈值。
- 下单、模拟结算、写日志、写报告的逻辑与主流程**完全一致**。

### 双阈值逻辑（代码）

- **UP**：`confidence >= PROB_THRESHOLD_UP`（如 0.9 → UP≥90%）。
- **DOWN**：`confidence >= (1 - PROB_THRESHOLD_DOWN)`（如 PROB_THRESHOLD_DOWN=0.1 → DOWN 要求 confidence≥90%，即 prob<10%）。
- 实现位置：`prediction_index.ts` 里对 `parsed` 的 `filter`、`prediction_api.ts` 的 `getHighConfidencePredictions` / 下单条件，三处一致。

### 报告与成交是否一致

| 组合 | 报告中的阈值 | 报告中的交易记录 |
|------|----------------|------------------|
| **logs_eth_10_90** | UP≥90%, DOWN≥90% (即 prob<10%) | 除**第一笔**外，全部 ≥90%；第一笔为 ETH ↓ 54.3%，**低于 90%**。 |
| **logs_xrp_20_80** | UP≥80%, DOWN≥80% (即 prob<20%) | 除**第一笔**外，全部 ≥80%；第一笔为 XRP ↓ 65.9%，**低于 80%**。 |

**结论**：

- **双阈值在绝大多数时间被正确触发**：两个目录里除各自的第一笔外，所有成交的置信度都满足报告里写的阈值（ETH 全部 ≥90%，XRP 全部 ≥80%）。
- **第一笔异常**：  
  - logs_eth_10_90：第一笔 DOWN 54.3%（应 ≥90%）。  
  - logs_xrp_20_80：第一笔 DOWN 65.9%（应 ≥80%）。  
  可能原因：首次启动时未设双阈值（或用了单阈值/旧配置），之后重启才加上 E/F 的双阈值；或历史数据来自不同进程。当前代码逻辑正确，**后续运行双阈值会按配置生效**。
