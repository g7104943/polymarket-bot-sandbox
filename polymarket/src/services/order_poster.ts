/**
 * 下单提交抽象：单钱包 / 多钱包并发（后续扩展）
 * 用于「监听到下单最快框架」：触发时只做 POST，不做签名，多钱包可并发 POST 同单。
 * 支持 FOK/FAK 市价单和 GTC/GTD 限价单。
 */

import type { ClobClient } from '@polymarket/clob-client-v2';
import { OrderType } from '@polymarket/clob-client-v2';

/** 支持的订单类型（市价 + 限价） */
export type SupportedOrderType = OrderType.FOK | OrderType.FAK | OrderType.GTC | OrderType.GTD;

/** 单笔 POST 结果（与 CLOB postOrder 返回一致） */
export interface PostOrderResult {
    success?: boolean;
    orderID?: string;
    [key: string]: unknown;
}

/**
 * 订单提交器：支持单钱包现在、多钱包并发后续扩展。
 * - 单钱包：postOrder(signedOrder) 一次 POST。
 * - 多钱包（后续）：postOrders(signedOrders[]) 并发 POST 多笔，减少被夹。
 */
export interface IOrderPoster {
    /** 提交单笔已签名订单（触发时调用，最快路径） */
    postOrder(signedOrder: unknown, orderType: SupportedOrderType): Promise<PostOrderResult>;
    /**
     * 并发提交多笔已签名订单（多钱包时用，后续实现）。
     * 默认实现：调一次 postOrder。
     */
    postOrders?(signedOrders: unknown[], orderType: SupportedOrderType): Promise<PostOrderResult[]>;
}

/** 单钱包实现：直接委托给 ClobClient.postOrder */
export class SingleWalletOrderPoster implements IOrderPoster {
    constructor(private readonly clobClient: ClobClient) {}

    async postOrder(signedOrder: unknown, orderType: SupportedOrderType): Promise<PostOrderResult> {
        return this.clobClient.postOrder(signedOrder as any, orderType);
    }
}
