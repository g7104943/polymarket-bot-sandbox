"""
预测 API：FastAPI，提供 GET /predict/{symbol}、GET /health，供 TS 脚本拉取预测。
"""

import os
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .predictor import predict, predict_all, SYMBOLS, TIMEFRAMES

app = FastAPI(title="LightGBM-FreqAI Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/{symbol}")
def get_predict(
    symbol: str,
    timeframe: str = Query("15m", description="15m | 1h | 4h"),
):
    """返回指定 symbol 在给定 timeframe 的下一根 K 线涨跌预测。"""
    symbol = symbol.upper()
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    if symbol not in SYMBOLS:
        raise HTTPException(400, f"不支持的 symbol，支持: {SYMBOLS}")
    if timeframe not in TIMEFRAMES:
        raise HTTPException(400, f"不支持的 timeframe，支持: {TIMEFRAMES}")
    try:
        direction, confidence = predict(symbol, timeframe)
        return {"symbol": symbol, "timeframe": timeframe, "direction": direction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/predict")
def get_predict_all(
    timeframes: Optional[str] = Query(None, description="逗号分隔，如 15m,1h,4h，默认全部"),
):
    """返回所有 symbol×timeframe 的预测。"""
    tfs = [s.strip() for s in timeframes.split(",")] if timeframes else list(TIMEFRAMES)
    tfs = [t for t in tfs if t in TIMEFRAMES]
    if not tfs:
        tfs = list(TIMEFRAMES)
    try:
        result = predict_all(timeframes=tfs)
        out = {}
        for (s, tf), v in result.items():
            key = f"{s}_{tf}"
            out[key] = v
        return {"predictions": out}
    except Exception as e:
        raise HTTPException(500, str(e))


def main():
    import uvicorn

    port = int(os.environ.get("MODEL_PREDICTION_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
