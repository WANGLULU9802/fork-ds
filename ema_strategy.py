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
from logger_config import setup_logging

# 设置日志系统 - 支持同时输出到控制台和文件
logger = setup_logging(
    log_filename='app.log',
    log_level=logging.INFO,
    name='ema_strategy'
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


# 创建交易所实例
exchange = ccxt.binance(config)

# 交易参数配置
TRADE_CONFIG = {
    'symbol': 'SOL/USDT',
    'base_currency': 'SOL',
    'amount': 0.001,  # 交易数量 (本位币)
    'leverage': 10,  # 杠杆倍数
    'timeframe': '15m',
    'high_timeframe': '15m',
    'test_mode': True,  # 测试模式
}

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None


def calculate_ema(prices, period):
    """
    计算EMA指数移动平均线

    Args:
        prices: 价格列表
        period: EMA周期

    Returns:
        float: EMA值
    """
    if len(prices) < period:
        return None

    # 计算SMA作为初始EMA
    initial_sma = sum(prices[:period]) / period

    # 计算平滑系数
    multiplier = 2 / (period + 1)

    # 计算EMA
    ema = initial_sma
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return ema


def calculate_rsi(prices, period=9):
    """
    计算RSI相对强弱指标

    Args:
        prices: 价格列表
        period: RSI周期，默认为9

    Returns:
        float: RSI值
    """
    if len(prices) < period + 1:
        return None

    # 计算价格变化
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

    # 分离涨跌
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]

    # 计算初始平均涨跌幅
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # 使用平滑公式计算后续的平均涨跌幅
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # 计算RS和RSI
    if avg_loss == 0:
        return 100  # 没有跌幅，RSI为100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_technical_indicators(price_data, calculate_historical=False):
    """
    计算技术指标（EMA21, EMA50, RSI9）

    Args:
        price_data: 当前价格数据或历史价格列表
        calculate_historical: 是否计算历史K线的技术指标

    Returns:
        dict or list: 技术指标数据
    """
    indicators = {}

    if calculate_historical:
        # 计算历史K线的技术指标
        historical_indicators = []

        # 获取所有历史价格
        all_prices = [data['price'] for data in price_history]
        logger.info(f"计算历史指标，价格数据数量: {len(all_prices)}")

        if len(all_prices) >= 50:
            # 为每根K线计算技术指标
            for i in range(len(price_history)):
                if i < 49:  # 前49根K线数据不足以计算EMA50
                    historical_indicators.append({
                        'ema21': None,
                        'ema50': None,
                        'rsi9': None,
                        'price_vs_ema21': None,
                        'price_vs_ema50': None,
                        'ema21_vs_ema50': None
                    })
                else:
                    # 获取到当前K线为止的价格数据
                    prices_so_far = all_prices[:i+1]
                    current_price = price_history[i]['price']

                    indicator = {}

                    # 计算EMA21
                    ema21 = calculate_ema(prices_so_far, 21)
                    if ema21:
                        indicator['ema21'] = ema21
                        indicator['price_vs_ema21'] = ((current_price - ema21) / ema21) * 100
                    else:
                        indicator['ema21'] = None
                        indicator['price_vs_ema21'] = None

                    # 计算EMA50
                    ema50 = calculate_ema(prices_so_far, 50)
                    if ema50:
                        indicator['ema50'] = ema50
                        indicator['price_vs_ema50'] = ((current_price - ema50) / ema50) * 100
                    else:
                        indicator['ema50'] = None
                        indicator['price_vs_ema50'] = None

                    # 计算RSI9 (需要至少10个价格点)
                    if len(prices_so_far) >= 10:
                        rsi9 = calculate_rsi(prices_so_far, 9)
                        indicator['rsi9'] = rsi9
                    else:
                        indicator['rsi9'] = None

                    # 计算EMA21和EMA50的相对位置
                    if ema21 and ema50:
                        indicator['ema21_vs_ema50'] = ((ema21 - ema50) / ema50) * 100
                    else:
                        indicator['ema21_vs_ema50'] = None

                    historical_indicators.append(indicator)

        return historical_indicators

    else:
        # 原有的当前价格指标计算逻辑
        closes = [data['price'] for data in price_history]

        if len(closes) >= 50:  # 需要足够的数据计算EMA50
            current_price = price_data['price']

            # 计算EMA21
            ema21 = calculate_ema(closes, 21)
            if ema21:
                indicators['ema21'] = ema21
                indicators['price_vs_ema21'] = ((current_price - ema21) / ema21) * 100

            # 计算EMA50
            ema50 = calculate_ema(closes, 50)
            if ema50:
                indicators['ema50'] = ema50
                indicators['price_vs_ema50'] = ((current_price - ema50) / ema50) * 100

            # 计算RSI9
            if len(closes) >= 10:  # 需要至少10个价格点计算RSI9
                rsi9 = calculate_rsi(closes, 9)
                indicators['rsi9'] = rsi9

            # 计算EMA21和EMA50的相对位置
            if ema21 and ema50:
                indicators['ema21_vs_ema50'] = ((ema21 - ema50) / ema50) * 100

        return indicators


def initialize_historical_data():
    """初始化历史数据"""
    global price_history

    try:
        # 获取60根K线作为初始历史数据
        logger.info("开始初始化历史数据...")
        initial_data = get_ohlcv(TRADE_CONFIG['timeframe'], initialize=True)

        if not initial_data or 'historical_prices' not in initial_data:
            logger.error("获取初始历史数据失败")
            return False

        # 构建历史数据列表
        price_history = []
        historical_prices = initial_data['historical_prices']

        for i, price in enumerate(historical_prices):
            price_point = {
                'price': price,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'high': 0,  # 初始化时不保留详细K线数据
                'low': 0,
                'volume': 0,
                'timeframe': TRADE_CONFIG['timeframe'],
                'price_change': 0,
                'kline_data': []
            }
            price_history.append(price_point)

        logger.info(f"成功初始化{len(price_history)}个历史数据点")
        return True

    except Exception as e:
        logger.exception(f"初始化历史数据失败: {e}")
        return False


def setup_exchange():
    """设置交易所参数"""
    try:
        # 设置杠杆
        exchange.set_leverage(TRADE_CONFIG['leverage'], TRADE_CONFIG['symbol'])
        logger.info(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")

        # 获取余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        logger.info(f"当前USDT余额: {usdt_balance:.2f}")

        return True
    except Exception as e:
        logger.exception(f"交易所设置失败: {e}")
        return False


def get_ohlcv(timeframe, initialize=False):
    """获取K线数据

    Args:
        timeframe: 时间周期
        initialize: 是否为初始化模式（获取更多历史数据）
    """
    try:
        # 根据是否为初始化模式决定获取的K线数量
        if initialize:
            limit = 60  # 初始化时获取60根K线确保足够计算EMA
        else:
            limit = 10  # 正常运行时获取10根K线

        # 添加网络请求重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], timeframe, limit=limit)
                break  # 成功获取数据，跳出重试循环
            except Exception as network_error:
                if attempt == max_retries - 1:  # 最后一次重试
                    raise network_error
                logger.warning(f"网络请求失败，第{attempt + 1}次重试: {network_error}")
                time.sleep(2 * (attempt + 1))  # 递增延迟

        # 转换为DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data

        result = {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': timeframe,
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
        }

        # 如果是初始化模式，添加所有历史价格数据
        if initialize:
            # 修复：使用 DataFrame 的 close 列，而不是 itertuples()
            result['historical_prices'] = df['close'].tolist()
            logger.info(f"初始化模式：获取了{len(result['historical_prices'])}根{timeframe}K线历史数据")

        return result
    except Exception as e:
        logger.exception(f"获取K线数据失败: {e}")
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

                logger.info(f"调试 - 持仓量: {position_amt}")

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

        logger.info("调试 - 未找到有效持仓")
        return None

    except Exception as e:
        logger.info(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_with_deepseek(price_data, high_price_data):
    """使用DeepSeek分析市场并生成交易信号"""

    # 更新历史数据（移除最旧的数据，添加最新的数据）
    if price_history:
        price_history.pop(0)  # 移除最旧的数据
        price_history.append(price_data)  # 添加最新数据

    # 计算历史K线的EMA指标
    historical_indicators = get_technical_indicators(None, calculate_historical=True)

    # 构建K线数据文本（包含EMA指标）
    kline_text = f"【由远到近的10根{TRADE_CONFIG['timeframe']}K线数据（含EMA指标）】\n"

    # 获取最近10根K线数据
    klines = price_data['kline_data']
    klines_count = len(klines)

    # 获取对应的EMA指标（如果有的话）
    if historical_indicators and len(historical_indicators) >= klines_count:
        # 使用最近计算的指标
        recent_indicators = historical_indicators[-klines_count:]
    else:
        # 指标数据不足，创建空指标
        recent_indicators = [None] * klines_count

    for i, kline in enumerate(klines):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100

        # 基本K线信息
        kline_info = f"K线{i + 1}: {trend} O:{kline['open']:.2f} C:{kline['close']:.2f} H:{kline['high']:.2f} L:{kline['low']:.2f} V:{kline['volume']:.2f} 涨跌:{change:+.2f}%"

        # 添加技术指标信息
        indicator = recent_indicators[i]
        if indicator and indicator['ema21'] and indicator['ema50']:
            kline_info += f" | EMA21:{indicator['ema21']:.2f} EMA50:{indicator['ema50']:.2f}"
            if indicator.get('rsi9'):
                kline_info += f" RSI9:{indicator['rsi9']:.2f}"
        elif indicator and indicator['ema21']:
            kline_info += f" | EMA21:{indicator['ema21']:.2f}"
            if indicator.get('rsi9'):
                kline_info += f" RSI9:{indicator['rsi9']:.2f}"
        else:
            kline_info += " | 技术指标数据不足"

        kline_text += kline_info + "\n"

  
    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"

    prompt = f"""
    你是一个专业的价格行为交易者。请基于以下{TRADE_CONFIG['symbol']} {TRADE_CONFIG['timeframe']}周期数据进行分析：

    {kline_text}

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
    1. 基于Albrooks的价格行为分析方法和{TRADE_CONFIG['timeframe']}K线趋势和技术指标给出交易信号: BUY(买入) / SELL(卖出) / HOLD(观望)
    2. 简要分析理由（重点考虑价格行为，EMA21和EMA50的关系、价格与EMA的位置）
    3. 基于EMA指标、RSI指标和支撑阻力分析建议合理的止损价位
    4. 基于EMA指标、RSI指标和阻力位分析建议合理的止盈价位
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
                 "content": f"您是一位急需资金为母亲治疗癌症的交易员，你现在非常非常非常需要从加密货币交易市场赚取足够的金额治疗你的母亲，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线价格行为、技术指标做出判断。"},
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
            logger.info(f"无法解析JSON: {result}")
            return None

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        return signal_data

    except Exception as e:
        logger.error(f"DeepSeek分析失败，原始文本: {result}")
        logger.exception(f"DeepSeek分析失败: {e}")
        return None


def execute_trade(signal_data, price_data):
    """执行交易（简化版）"""
    current_position = get_current_position()

    logger.info(f"交易信号: {signal_data['signal']}")
    logger.info(f"信心程度: {signal_data['confidence']}")
    logger.info(f"止损价格: {signal_data['stop_loss']}")
    logger.info(f"理由: {signal_data['reason']}")
    logger.info(f"当前持仓: {current_position}")

    if TRADE_CONFIG['test_mode']:
        logger.info("测试模式 - 仅模拟交易")
        return

    try:
        # 简化的交易逻辑：只处理单向持仓
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                # 平空仓
                logger.info("平空仓...")
                exchange.create_market_buy_order(
                    TRADE_CONFIG['symbol'],
                    current_position['size'],
                    {'posSide': 'short'}
                )
            elif not current_position or current_position['side'] == 'long':
                # 开多仓或加多仓
                logger.info("开多仓...")
                exchange.create_market_buy_order(
                    TRADE_CONFIG['symbol'],
                    TRADE_CONFIG['amount'],
                    {'posSide': 'long'}
                )

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                # 平多仓
                logger.info("平多仓...")
                exchange.create_market_sell_order(
                    TRADE_CONFIG['symbol'],
                    current_position['size'],
                    {'posSide': 'long'}
                )
            elif not current_position or current_position['side'] == 'short':
                # 开空仓或加空仓
                logger.info("开空仓...")
                exchange.create_market_sell_order(
                    TRADE_CONFIG['symbol'],
                    TRADE_CONFIG['amount'],
                    {'posSide': 'short'}
                )

        elif signal_data['signal'] == 'HOLD':
            logger.info("建议观望，不执行交易")
            return

        logger.info("订单执行成功")
        time.sleep(2)
        position = get_current_position()
        logger.info(f"更新后持仓: {position}")

    except Exception as e:
        logger.info(f"订单执行失败: {e}")
        import traceback
        traceback.print_exc()

def trading_bot():
    """主交易机器人函数"""
    logger.info("\n" + "=" * 60)
    logger.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. 获取K线数据
    price_data = get_ohlcv(TRADE_CONFIG['timeframe'])
    high_price_data = get_ohlcv(TRADE_CONFIG['high_timeframe'])

    if not price_data:
        return

    logger.info(f"{TRADE_CONFIG['base_currency']}当前价格: ${price_data['price']:,.2f}")
    logger.info(f"数据周期: {TRADE_CONFIG['timeframe']}")
    logger.info(f"价格变化: {price_data['price_change']:+.2f}%")

    # 计算并显示技术指标
    indicators = get_technical_indicators(price_data)
    if indicators:
        if 'ema21' in indicators:
            logger.info(f"EMA21: ${indicators['ema21']:.2f} (价格相对: {indicators['price_vs_ema21']:+.2f}%)")
        if 'ema50' in indicators:
            logger.info(f"EMA50: ${indicators['ema50']:.2f} (价格相对: {indicators['price_vs_ema50']:+.2f}%)")
        if 'rsi9' in indicators:
            rsi_status = ""
            if indicators['rsi9'] > 70:
                rsi_status = " (超买)"
            elif indicators['rsi9'] < 30:
                rsi_status = " (超卖)"
            logger.info(f"RSI9: {indicators['rsi9']:.2f}{rsi_status}")
        if 'ema21_vs_ema50' in indicators:
            if indicators['ema21_vs_ema50'] > 0:
                trend = "看涨"
            else:
                trend = "看跌"
            logger.info(f"EMA趋势: {trend} (EMA21相对EMA50: {indicators['ema21_vs_ema50']:+.2f}%)")
    else:
        logger.info("技术指标: 计算失败")

    # 2. 使用DeepSeek分析
    signal_data = analyze_with_deepseek(price_data, high_price_data)
    if not signal_data:
        return

    # 3. 执行交易
    execute_trade(signal_data, price_data)


def main():
    """主函数"""
    logger.info(f"{TRADE_CONFIG['symbol']}自动交易机器人启动成功！")

    if TRADE_CONFIG['test_mode']:
        logger.info("当前为模拟模式，不会真实下单")
    else:
        logger.info("实盘交易模式，请谨慎操作！")

    logger.info(f"交易周期: {TRADE_CONFIG['timeframe']}")
    logger.info("已启用K线数据分析和持仓跟踪功能")

    # 设置交易所
    if not setup_exchange():
        logger.info("交易所初始化失败，程序退出")
        return

    # 初始化历史数据
    if not initialize_historical_data():
        logger.info("历史数据初始化失败，程序退出")
        return

    # 验证技术指标计算
    test_price_data = {'price': price_history[-1]['price'] if price_history else 0}
    test_indicators = get_technical_indicators(test_price_data)
    if test_indicators:
        logger.info("技术指标验证成功:")
        if 'ema21' in test_indicators:
            logger.info(f"  EMA21: ${test_indicators['ema21']:.2f}")
        if 'ema50' in test_indicators:
            logger.info(f"  EMA50: ${test_indicators['ema50']:.2f}")
        if 'rsi9' in test_indicators:
            logger.info(f"  RSI9: {test_indicators['rsi9']:.2f}")
        if 'ema21_vs_ema50' in test_indicators:
            logger.info(f"  EMA关系: {test_indicators['ema21_vs_ema50']:+.2f}%")
    else:
        logger.info("警告: 技术指标验证失败")

    # 根据时间周期设置执行频率
    if TRADE_CONFIG['timeframe'] == '1h':
        # 每小时执行一次，在整点后的1分钟执行
        schedule.every().hour.at(":01").do(trading_bot)
        logger.info("执行频率: 每小时一次")
    elif TRADE_CONFIG['timeframe'] == '15m':
        # 每15分钟执行一次
        schedule.every(15).minutes.do(trading_bot)
        logger.info("执行频率: 每15分钟一次")
    elif TRADE_CONFIG['timeframe'] == '5m':
        # 每15分钟执行一次
        schedule.every(5).minutes.do(trading_bot)
        logger.info("执行频率: 每5分钟一次")
    else:
        # 默认1小时
        schedule.every().hour.at(":01").do(trading_bot)
        logger.info("执行频率: 每小时一次")

    # 立即执行一次
    trading_bot()

    # 循环执行
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()