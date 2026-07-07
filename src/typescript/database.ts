/**
 * MongoDB 存储交易记录与预测记录
 */

import { MongoClient, Db, Collection } from "mongodb";
import { getEnv } from "./utils";

let client: MongoClient | null = null;
let db: Db | null = null;

const MONGO_URI = "MONGO_URI";
const DB_NAME = "polymarket_trading";

export interface TradeRecord {
  _id?: string;
  symbol: string;
  timeframe: string;
  direction: "UP" | "DOWN";
  confidence: number;
  amount: number;
  outcome: "YES" | "NO";
  marketId?: string;
  txHash?: string;
  mode: "simulation" | "live";
  createdAt: Date;
  result?: "won" | "lost" | "pending";
}

export interface PredictionRecord {
  _id?: string;
  symbol: string;
  timeframe: string;
  direction: "UP" | "DOWN";
  confidence: number;
  receivedAt: Date;
}

export async function getDb(): Promise<Db> {
  if (db) return db;
  const uri = getEnv(MONGO_URI, "mongodb://localhost:27017");
  client = new MongoClient(uri);
  await client.connect();
  db = client.db(DB_NAME);
  return db;
}

export async function getTradesCollection(): Promise<Collection<TradeRecord>> {
  const d = await getDb();
  return d.collection<TradeRecord>("trades");
}

export async function getPredictionsCollection(): Promise<Collection<PredictionRecord>> {
  const d = await getDb();
  return d.collection<PredictionRecord>("predictions");
}

export async function insertTrade(record: Omit<TradeRecord, "_id" | "createdAt">): Promise<void> {
  const col = await getTradesCollection();
  await col.insertOne({
    ...record,
    createdAt: new Date(),
  } as TradeRecord);
}

export async function insertPrediction(
  record: Omit<PredictionRecord, "_id" | "receivedAt">
): Promise<void> {
  const col = await getPredictionsCollection();
  await col.insertOne({
    ...record,
    receivedAt: new Date(),
  } as PredictionRecord);
}

export async function closeDb(): Promise<void> {
  if (client) {
    await client.close();
    client = null;
    db = null;
  }
}
