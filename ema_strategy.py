import os
import platform
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
from datetime import datetime
import json
from dotenv import load_dotenv
import logging
import re


# 配置日志
logging.basicConfig(
    filename='app.log',  # 日志文件名
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # 支持中文
)

load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)


# 判断是否为 Windows 系统
is_windows = platform.system() == 'Windows'

# 配置交易所参数
config = {
    'options': {'defaultType': 'future'},
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET'),
}

# 如果是 Windows 系统，添加代理配置
if is_windows:
    # 方法1：使用 proxies 参数（推荐）
    config['proxies'] = {
        'http': 'http://127.0.0.1:7890',  # 替换为你的代理地址
        'https': 'http://127.0.0.1:7890',  # 替换为你的代理地址
    }

    # 或者方法2：使用单独的代理参数
    # config['httpProxy'] = 'http://127.0.0.1:7890'
    # config['httpsProxy'] = 'http://127.0.0.1:7890'

# 创建交易所实例
exchange = ccxt.binance(config)

# 交易参数配置
TRADE_CONFIG = {
    'symbol': 'SOL/USDT',
    'base_currency': 'SOL',
    'amount': 0.001,  # 交易数量 (本位币)
    'leverage': 10,  # 杠杆倍数
    'timeframe': '5m', 
    'high_timeframe': '15m',
    'test_mode': True,  # 测试模式
}

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None


def setup_exchange():
    """设置交易所参数"""
    try:
        # 设置杠杆
        exchange.set_leverage(TRADE_CONFIG['leverage'], TRADE_CONFIG['symbol'])
        logging.info(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")

        # 获取余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        logging.info(f"当前USDT余额: {usdt_balance:.2f}")

        return True
    except Exception as e:
        logging.exception(f"交易所设置失败: {e}")
        return False


def get_ohlcv(timeframe):
    """获取K线数据（1小时或15分钟）"""
    try:
        # 获取最近10根K线
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], timeframe, limit=10)

        # 转换为DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': timeframe,
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
        }
    except Exception as e:
        logging.info(f"获取K线数据失败: {e}")
        return None


def get_current_position():
    """获取当前持仓情况"""
    try:
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        # 标准化配置的交易对符号用于比较
        config_symbol_normalized = f"{TRADE_CONFIG['symbol']}:USDT"

        for pos in positions:

            # 比较标准化的符号
            if pos['symbol'] == config_symbol_normalized:
                # 获取持仓数量
                position_amt = 0
                if 'positionAmt' in pos.get('info', {}):
                    position_amt = float(pos['info']['positionAmt'])
                elif 'contracts' in pos:
                    # 使用 contracts 字段，根据 side 确定方向
                    contracts = float(pos['contracts'])
                    if pos.get('side') == 'short':
                        position_amt = -contracts
                    else:
                        position_amt = contracts

                logging.info(f"调试 - 持仓量: {position_amt}")

                if position_amt != 0:  # 有持仓
                    side = 'long' if position_amt > 0 else 'short'
                    return {
                        'side': side,
                        'size': abs(position_amt),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                        'position_amt': position_amt,
                        'symbol': pos['symbol']  # 返回实际的symbol用于调试
                    }

        logging.info("调试 - 未找到有效持仓")
        return None

    except Exception as e:
        logging.info(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_with_deepseek(price_data, high_price_data):
    """使用DeepSeek分析市场并生成交易信号"""

    # 添加当前价格到历史记录
    price_history.append(price_data)
    if len(price_history) > 20:  # 保留更多历史数据用于长周期分析
        price_history.pop(0)

    # 构建K线数据文本
    kline_text = f"【由远到近的10根{TRADE_CONFIG['timeframe']}K线数据】\n"
    for i, kline in enumerate(price_data['kline_data']):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"K线{i + 1}: {trend} O:{kline['open']:.2f} C:{kline['close']:.2f} H:{kline['high']:.2f} L:{kline['low']:.2f} V:{kline['volume']:.2f} 涨跌:{change:+.2f}%\n"

    kline_text += f"【由远到近的10根{TRADE_CONFIG['high_timeframe']}K线数据】\n"
    for i, kline in enumerate(high_price_data['kline_data']):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"K线{i + 1}: {trend} O:{kline['open']:.2f} C:{kline['close']:.2f} H:{kline['high']:.2f} L:{kline['low']:.2f} V:{kline['volume']:.2f} 涨跌:{change:+.2f}%\n"



    # 构建技术指标文本
    if len(price_history) >= 5:
        closes = [data['price'] for data in price_history[-5:]]
        sma_5 = sum(closes) / len(closes)
        price_vs_sma = ((price_data['price'] - sma_5) / sma_5) * 100

        indicator_text = f"【技术指标】\n5周期均价: {sma_5:.2f}\n当前价格相对于均线: {price_vs_sma:+.2f}%"
    else:
        indicator_text = "【技术指标】\n数据不足计算技术指标"

    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下{TRADE_CONFIG['symbol']} {TRADE_CONFIG['timeframe']}周期数据进行分析：

    {kline_text}

    {indicator_text}

    {signal_text}

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} {TRADE_CONFIG['base_currency']}
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}

    【分析要求】
    1. 基于{TRADE_CONFIG['timeframe']}K线趋势和技术指标给出交易信号: BUY(买入) / SELL(卖出) / HOLD(观望)
    2. 简要分析理由（考虑趋势连续性、支撑阻力、成交量等因素）
    3. 基于技术分析建议合理的止损价位
    4. 基于技术分析建议合理的止盈价位
    5. 评估信号信心程度

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "分析理由",
        "stop_loss": 具体价格,
        "take_profit": 具体价格,
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    """

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"您是一位急需资金为母亲治疗癌症的交易员，你现在非常非常非常需要从加密货币交易市场赚取足够的金额治疗你的母亲，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            # 🔧 关键修复：移除数字中的逗号
            json_str = re.sub(r':\s*(\d{1,3}(,\d{3})+)', lambda m: ': ' + m.group(1).replace(',', ''), json_str)
        
            signal_data = json.loads(json_str)
        else:
            logging.info(f"无法解析JSON: {result}")
            return None

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        return signal_data

    except Exception as e:
        logging.error(f"DeepSeek分析失败，原始文本: {result}")
        logging.exception(f"DeepSeek分析失败: {e}")
        return None


def execute_trade(signal_data, price_data):
    """执行交易（简化版）"""
    current_position = get_current_position()

    logging.info(f"交易信号: {signal_data['signal']}")
    logging.info(f"信心程度: {signal_data['confidence']}")
    logging.info(f"止损价格: {signal_data['stop_loss']}")
    logging.info(f"理由: {signal_data['reason']}")
    logging.info(f"当前持仓: {current_position}")

    if TRADE_CONFIG['test_mode']:
        logging.info("测试模式 - 仅模拟交易")
        return

    try:
        # 简化的交易逻辑：只处理单向持仓
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                # 平空仓
                logging.info("平空仓...")
                exchange.create_market_buy_order(
                    TRADE_CONFIG['symbol'],
                    current_position['size'],
                    {'posSide': 'short'}
                )
            elif not current_position or current_position['side'] == 'long':
                # 开多仓或加多仓
                logging.info("开多仓...")
                exchange.create_market_buy_order(
                    TRADE_CONFIG['symbol'],
                    TRADE_CONFIG['amount'],
                    {'posSide': 'long'}
                )

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                # 平多仓
                logging.info("平多仓...")
                exchange.create_market_sell_order(
                    TRADE_CONFIG['symbol'],
                    current_position['size'],
                    {'posSide': 'long'}
                )
            elif not current_position or current_position['side'] == 'short':
                # 开空仓或加空仓
                logging.info("开空仓...")
                exchange.create_market_sell_order(
                    TRADE_CONFIG['symbol'],
                    TRADE_CONFIG['amount'],
                    {'posSide': 'short'}
                )

        elif signal_data['signal'] == 'HOLD':
            logging.info("建议观望，不执行交易")
            return

        logging.info("订单执行成功")
        time.sleep(2)
        position = get_current_position()
        logging.info(f"更新后持仓: {position}")

    except Exception as e:
        logging.info(f"订单执行失败: {e}")
        import traceback
        traceback.print_exc()

def trading_bot():
    """主交易机器人函数"""
    logging.info("\n" + "=" * 60)
    logging.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)

    # 1. 获取K线数据
    price_data = get_ohlcv(TRADE_CONFIG['timeframe'])
    high_price_data = get_ohlcv(TRADE_CONFIG['high_timeframe'])

    if not price_data:
        return

    logging.info(f"{TRADE_CONFIG['base_currency']}当前价格: ${price_data['price']:,.2f}")
    logging.info(f"数据周期: {TRADE_CONFIG['timeframe']}")
    logging.info(f"价格变化: {price_data['price_change']:+.2f}%")

    # 2. 使用DeepSeek分析
    signal_data = analyze_with_deepseek(price_data, high_price_data)
    if not signal_data:
        return

    # 3. 执行交易
    execute_trade(signal_data, price_data)


def main():
    """主函数"""
    logging.info(f"{TRADE_CONFIG['symbol']}自动交易机器人启动成功！")

    if TRADE_CONFIG['test_mode']:
        logging.info("当前为模拟模式，不会真实下单")
    else:
        logging.info("实盘交易模式，请谨慎操作！")

    logging.info(f"交易周期: {TRADE_CONFIG['timeframe']}")
    logging.info("已启用K线数据分析和持仓跟踪功能")

    # 设置交易所
    if not setup_exchange():
        logging.info("交易所初始化失败，程序退出")
        return

    # 根据时间周期设置执行频率
    if TRADE_CONFIG['timeframe'] == '1h':
        # 每小时执行一次，在整点后的1分钟执行
        schedule.every().hour.at(":01").do(trading_bot)
        logging.info("执行频率: 每小时一次")
    elif TRADE_CONFIG['timeframe'] == '15m':
        # 每15分钟执行一次
        schedule.every(15).minutes.do(trading_bot)
        logging.info("执行频率: 每15分钟一次")
    elif TRADE_CONFIG['timeframe'] == '5m':
        # 每15分钟执行一次
        schedule.every(5).minutes.do(trading_bot)
        logging.info("执行频率: 每5分钟一次")
    else:
        # 默认1小时
        schedule.every().hour.at(":01").do(trading_bot)
        logging.info("执行频率: 每小时一次")

    # 立即执行一次
    trading_bot()

    # 循环执行
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()