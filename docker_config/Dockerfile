# syntax=docker/dockerfile:1
# 使用包含强化学习库的标准FreqAI镜像
FROM freqtradeorg/freqtrade:develop_freqairl

# 切换到root用户以安装系统依赖
USER root

# 1. 安装必要的编译工具和系统库
RUN apt-get update \
  && apt-get -y install --no-install-recommends build-essential libssl-dev git libffi-dev \
  && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# --- ✅ 优化：移除多余的PyTorch安装步骤 ---
# 基础镜像 freqtradeorg/freqtrade:develop_freqairl 已包含PyTorch。
# 无需再手动执行 pip install torch。
# ---------------------------------------------------------------------------

# 2. 复制所有用户文件到容器中
#    Docker会自动处理目录创建，这种方式更简洁高效
#COPY user_data/ /freqtrade/user_data/

# 3. 修复文件权限
#    将所有user_data下的文件所有权交给ftuser，解决Permission denied问题
RUN chown -R ftuser:ftuser /freqtrade/user_data

# 4. 切换回普通用户ftuser，增加安全性
USER ftuser

# 5. 设置工作目录
WORKDIR /freqtrade

# 6. 安装freqtrade本身和其他依赖
#    pip install -e . 会以可编辑模式安装，让所有路径和模块都能被正确找到
#    这比手动设置PYTHONPATH更健壮、更标准
RUN pip install -e .

# 7. 安装Kronos所需依赖并预缓存Kronos-Tokenizer-base
#    - 安装transformers相关依赖
#    - 设置HF缓存目录到容器内可挂载路径，便于复用与离线启动
#    - 预下载分词器以加速首次启动
ENV HF_HOME=/freqtrade/.cache/huggingface \
    TRANSFORMERS_CACHE=/freqtrade/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/freqtrade/.cache/huggingface/datasets

RUN pip install --no-cache-dir \
    transformers \
    tokenizers \
    accelerate \
    safetensors \
    huggingface-hub

RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base'); print('Kronos-Tokenizer cached')"
