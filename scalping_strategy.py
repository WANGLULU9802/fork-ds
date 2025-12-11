import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from openai import OpenAI
import time
from config.logger_config import setup_logging
from dotenv import load_dotenv
import pytz

load_dotenv()

# 东八区时区
beijing_tz = pytz.timezone('Asia/Shanghai')

def get_beijing_time():
    """获取东八区当前时间"""
    return datetime.now(beijing_tz)

def is_5min_interval():
    """检查当前时间是否是5分钟的整点"""
    now = get_beijing_time()
    minute = now.minute
    second = now.second

    # 检查是否是5分钟的整点（0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55）
    # 并且秒数小于10，确保在整点附近执行
    return minute % 5 == 0 and second < 10

def wait_until_next_5min():
    """等待到下一个5分钟整点"""
    now = get_beijing_time()
    minute = now.minute
    second = now.second

    # 计算到下一个5分钟整点需要等待的秒数
    if minute % 5 == 0 and second < 10:
        # 当前正好是5分钟整点，直接执行
        wait_seconds = 0
    else:
        next_5min = ((minute // 5) + 1) * 5
        if next_5min >= 60:
            next_5min = 0
            wait_seconds = (60 - minute) * 60 - second
        else:
            wait_seconds = (next_5min - minute) * 60 - second

        if wait_seconds > 300:  # 防止等待时间过长
            wait_seconds = 0

    if wait_seconds > 0:
        next_time = (now + timedelta(seconds=wait_seconds)).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"等待到下一个5分钟整点（{next_time}），预计等待{wait_seconds}秒...")
        time.sleep(wait_seconds)

# 设置日志系统 - 支持同时输出到控制台和文件
logger = setup_logging(
    log_filename='app.log',
    log_level=logging.INFO,
    name='ema_strategy'
)


class ScalpingStrategy:
    def __init__(self, symbol='SOL/USDT', timeframe='5m', length=10):
        """
        初始化剥头皮策略

        Args:
            symbol: 交易对，默认SOL/USDT
            timeframe: 时间周期，默认5分钟
            length: 转折点识别周期，默认10
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.length = length

        # 初始化交易所
        self.exchange = ccxt.binance({
            'options': {'defaultType': 'future'},
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
        })

        # Windows代理配置
        if os.name == 'nt':
            self.exchange.proxies = {
                'http': 'http://127.0.0.1:7890',
                'https': 'http://127.0.0.1:7890',
            }

        # 初始化DeepSeek客户端
        self.deepseek_client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

        # 存储历史数据
        self.price_data = []
        self.pivot_points = []
        self.labels = []

    def fetch_ohlcv(self, limit=50):
        """获取K线数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # 转换为东八区时间
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=8)
            return df
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return None

    def calculate_pivots(self, df):
        """
        计算转折点 (类似Pine脚本中的pivots函数)

        Args:
            df: K线数据DataFrame

        Returns:
            DataFrame: 添加了转折点标记的数据
        """
        df = df.copy()
        df['ph'] = 0  # potential high pivot
        df['pl'] = 0  # potential low pivot
        df['pivot_high'] = np.nan
        df['pivot_low'] = np.nan

        for i in range(self.length, len(df) - self.length):
            # 检查是否为高点转折 (length周期内最高点)
            if df.iloc[i]['high'] == df.iloc[i-self.length:i+self.length+1]['high'].max():
                df.loc[i, 'ph'] = 1
                df.loc[i, 'pivot_high'] = df.iloc[i]['high']

            # 检查是否为低点转折 (length周期内最低点)
            if df.iloc[i]['low'] == df.iloc[i-self.length:i+self.length+1]['low'].min():
                df.loc[i, 'pl'] = 1
                df.loc[i, 'pivot_low'] = df.iloc[i]['low']

        # 对于最近的K线，使用可用的数据进行转折点计算
        for i in range(len(df) - self.length, len(df)):
            # 计算实际可用的范围
            start_idx = max(0, i - self.length)
            end_idx = min(len(df) - 1, i + self.length)

            # 只有当有足够的数据时才进行计算
            if end_idx - start_idx >= self.length:
                # 检查是否为高点转折
                if df.iloc[i]['high'] == df.iloc[start_idx:end_idx+1]['high'].max():
                    df.loc[i, 'ph'] = 1
                    df.loc[i, 'pivot_high'] = df.iloc[i]['high']

                # 检查是否为低点转折
                if df.iloc[i]['low'] == df.iloc[start_idx:end_idx+1]['low'].min():
                    df.loc[i, 'pl'] = 1
                    df.loc[i, 'pivot_low'] = df.iloc[i]['low']

        return df

    def calculate_atr(self, df, period=14):
        """计算ATR (Average True Range)"""
        df = df.copy()
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))

        df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()

        return df

    def identify_pivot_sequence(self, df):
        """
        识别转折点序列并标记HH/LH/HL/LL

        Args:
            df: 包含转折点的DataFrame

        Returns:
            DataFrame: 添加了标签的数据
        """
        df = df.copy()

        # 获取所有有效转折点
        pivots = []
        for i in range(len(df)):
            if not pd.isna(df.iloc[i]['pivot_high']):
                pivots.append({'index': i, 'value': df.iloc[i]['pivot_high'], 'type': 'high'})
            elif not pd.isna(df.iloc[i]['pivot_low']):
                pivots.append({'index': i, 'value': df.iloc[i]['pivot_low'], 'type': 'low'})

        # 标记HH/LH/HL/LL
        for i, pivot in enumerate(pivots):
            if i >= 2 and pivot['type'] == pivots[i-2]['type']:
                current_value = pivot['value']
                prev_value = pivots[i-2]['value']

                if pivot['type'] == 'high':
                    if current_value > prev_value:
                        label = 'HH'  # Higher High
                    else:
                        label = 'LH'  # Lower High
                else:  # low
                    if current_value > prev_value:
                        label = 'HL'  # Higher Low
                    else:
                        label = 'LL'  # Lower Low

                # 在图表上标记标签
                df.loc[pivot['index'], 'label'] = label
                df.loc[pivot['index'], 'label_value'] = pivot['value']

        return df, pivots

    def check_entry_conditions(self, df, current_bar_index):
        """
        简化的入场条件检查 - 只收集数据给AI分析
        不再进行手动信号判断，完全交给AI处理

        Args:
            df: 完整的K线数据
            current_bar_index: 当前K线索引

        Returns:
            dict: 包含所有必要数据的信息
        """
        # 获取最近30根K线数据用于AI分析
        recent_bars = df.iloc[max(0, current_bar_index-29):current_bar_index+1]

        # 获取最近6根K线内的标签信息
        recent_6_bars = df.iloc[max(0, current_bar_index-5):current_bar_index+1]
        recent_6_labels = recent_6_bars[recent_6_bars['label'].notna()]

        # 获取所有标签信息（用于AI分析）- 这30根K线内的所有标签
        all_labels = recent_bars[recent_bars['label'].notna()]

        # 准备给AI的数据
        data_for_ai = {
            'df': recent_bars,
            'labels': all_labels,
            'recent_6_labels': recent_6_labels,
            'current_index': current_bar_index,
            'has_labels': len(recent_6_labels) > 0  # 只检查最近6根K线是否有标签
        }

        return data_for_ai

    def analyze_with_ai(self, data_for_ai, df):
        """
        使用AI分析市场数据和剥头皮策略机会

        Args:
            data_for_ai: 包含K线数据和标签信息的数据
            df: 完整的K线数据

        Returns:
            dict: AI分析结果
        """
        if not data_for_ai['has_labels']:
            logger.info("最近6根K线内没有发现标签，不进行AI分析，等待信号出现")
            return None

        recent_bars = data_for_ai['df']
        labels = data_for_ai['labels']

        # 构建K线文本，包含技术指标
        kline_text = f"最近30根{self.timeframe}K线数据及指标：\n"
        for i, (_, bar) in enumerate(recent_bars.iterrows()):
            change = ((bar['close'] - bar['open']) / bar['open']) * 100

            # 获取ATR值
            atr = bar['atr'] if 'atr' in bar and not pd.isna(bar['atr']) else 0

            # 获取EMA20值
            ema20 = bar['close'] * 0.9 if 'ema20' not in bar else bar['ema20']

            # 检查是否有标签
            label_info = ""
            if not pd.isna(bar['label']):
                label_info = f" 标签:{bar['label']}"

            kline_text += f"K{i+1}: O:{bar['open']:.2f} C:{bar['close']:.2f} H:{bar['high']:.2f} L:{bar['low']:.2f} V:{bar['volume']:.0f} 涨跌:{change:+.2f}% EMA20:{ema20:.2f} ATR:{atr:.4f}{label_info}\n"

        # 构建标签信息
        label_text = "发现的标签信息：\n"
        for idx, label_bar in labels.iterrows():
            # 计算这是第多少根K线（从最近30根的开始算起）
            k_index = 30 - len(recent_bars) + recent_bars.index.get_loc(idx) + 1
            label_text += f"- K{k_index}: 标签 {label_bar['label']} 价格: {label_bar['label_value']:.2f}\n"

        # 特别标注最近6根K线内的标签
        recent_6_labels = data_for_ai['recent_6_labels']
        if not recent_6_labels.empty:
            label_text += "\n最近6根K线内的标签（重点关注的信号）：\n"
            for idx, label_bar in recent_6_labels.iterrows():
                k_index = 30 - len(recent_bars) + recent_bars.index.get_loc(idx) + 1
                label_text += f"- K{k_index}: 标签 {label_bar['label']} 价格: {label_bar['label_value']:.2f}\n"

        prompt = f"""
你是一个专业的加密货币剥头皮交易员。

【剥头皮策略入场逻辑】
重要：标签K本身不是入场K线，标签K后5根K线才是入场K线！

做多入场逻辑：
1. 出现HL或LL标签（6K以内）
2. 标签K后5根K线中，有K线收盘价突破标签K最高点
3. 该入场K线不能有长上引线
4. 该入场K线大小不能超过ATR两倍
5. 止损设在标签K最低点，盈亏比0.5:1

做空入场逻辑：
1. 出现HH或LH标签（6K以内）
2. 标签K后5根K线中，有K线收盘价跌破标签K最低点
3. 该入场K线不能有长下引线
4. 该入场K线大小不能超过ATR两倍
5. 止损设在标签K最高点，盈亏比0.5:1

【市场数据】
交易对: {self.symbol}
时间周期: {self.timeframe}

{kline_text}

{label_text}

【分析任务】
请分析以上K线数据，识别各种价格结构：
1. K线形态（锤子线、十字星、吞没形态等）
2. 反转形态（双顶/底、头肩形、楔形等）
3. 持续形态（三角形、旗形、矩形等）
4. 缺口和测量距离
5. 趋势通道和交易区间
6. 动能反转信号

【分析要点】
请按以下顺序分析：
1. 找到最近6根K线内的标签（HH/LH/HL/LL）
2. 检查标签K后5根K线是否有符合条件的收盘价突破
3. 验证入场K线的引线大小和K线实体大小
4. 计算止损位和风险回报比

【关键判断】
- 标签K后5根K线内是否有符合条件的入场信号？
- 入场K线是否满足引线大小和ATR要求？
- 当前时机是否符合0.5:1的盈亏比要求？

【重要提醒】
- 标签K本身不能入场，必须等标签后5根K线
- 只有最近6根K线内的标签才考虑，超出时间的不做分析
- 入场K线收盘价必须突破标签K的关键价位
- 严格遵守引线大小和ATR过滤条件

请用以下JSON格式回复：
{{
    "recommendation": "BUY|SELL|SKIP|WAIT",
    "confidence": "HIGH|MEDIUM|LOW",
    "reason": "详细分析理由，说明标签K后5根K线内是否有符合条件的入场信号",
    "signal_details": {{
        "direction": "BUY|SELL|NONE",
        "label_type": "HL|LL|HH|LH|NONE",
        "label_k_position": "标签K的位置",
        "entry_k_position": "入场K的位置",
        "entry_price": "入场K收盘价",
        "stop_loss": "标签K止损位",
        "take_profit": "止盈位",
        "time_since_label": "标签后入场间隔",
        "entry_bar_valid": "入场K是否符合要求"
    }},
    "risk_assessment": "风险评估",
    "market_context": "市场背景分析",
    "entry_logic": "入场逻辑分析（重点说明标签K后5根K线的突破）",
    "label_timing": "标签时间有效性（6K内）"
}}
"""

        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"你是一个专业的加密货币剥头皮交易员，专注于{self.timeframe}周期的短期交易机会，擅长识别高风险回报的交易时机。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            result = response.choices[0].message.content
            # 提取JSON
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = result[start_idx:end_idx]
                ai_analysis = json.loads(json_str)
                return ai_analysis

        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return None

    def run_analysis(self):
        """运行简化的策略分析"""
        logger.info(f"开始分析 {self.symbol} {self.timeframe} 剥头皮策略")

        # 获取K线数据
        df = self.fetch_ohlcv(200)
        if df is None:
            logger.error("无法获取K线数据")
            return

        logger.info(f"获取到 {len(df)} 根K线数据，用于计算转折点和标签")

        # 计算转折点
        df = self.calculate_pivots(df)

        # 计算ATR
        df = self.calculate_atr(df)

        # 识别转折点序列并标记
        df, pivots = self.identify_pivot_sequence(df)

        # 获取数据给AI分析
        current_index = len(df) - 1  # 当前最新K线
        data_for_ai = self.check_entry_conditions(df, current_index)

        # 始终调用AI进行分析，让AI判断是否有信号
        ai_result = self.analyze_with_ai(data_for_ai, df)

        if ai_result:
            current_time = get_beijing_time().strftime('%Y-%m-%d %H:%M:%S')
            logger.info("=== AI分析结果 ===")
            logger.info(f"分析时间（东八区）: {current_time}")
            logger.info(f"建议: {ai_result['recommendation']}")
            logger.info(f"信心: {ai_result['confidence']}")
            logger.info(f"理由: {ai_result['reason']}")
            if ai_result.get('label_analysis'):
                logger.info(f"标签分析: {ai_result['label_analysis']}")

            return {
                'data_for_ai': data_for_ai,
                'ai_analysis': ai_result,
                'recommendation': ai_result['recommendation']
            }

        return None

def main():
    """主函数"""
    strategy = ScalpingStrategy(
        symbol='SOL/USDC',
        timeframe='5m',
        length=10
    )

    current_time = get_beijing_time().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"=== 剥头皮策略启动（东八区时间：{current_time}）===")
    logger.info("策略将收集30根K线数据和指标，交给AI进行信号判断")
    logger.info("AI将根据以下规则判断：")
    logger.info("- 做多：HL/LL标签出现，无长上引线，收盘突破标签最高点")
    logger.info("- 做空：HH/LH标签出现，无长下引线，收盘跌破标签最低点")
    logger.info("- 盈亏比：0.5:1")
    logger.info("- K线大小不超过ATR两倍")

    logger.info("策略将在5分钟整点自动运行分析（如: 04:00, 04:05, 04:10等）")

    while True:
        try:
            # 等待到下一个5分钟整点
            wait_until_next_5min()

            # 在5分钟整点执行分析
            current_time = get_beijing_time().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"=== 开始5分钟整点分析（{current_time}） ===")

            result = strategy.run_analysis()

            if result and result['recommendation'] in ['BUY', 'SELL']:
                logger.info("🚨 AI建议入场！")
                # 这里可以添加实际的交易执行逻辑
                signal_details = result['ai_analysis'].get('signal_details', {})
                if signal_details:
                    logger.info(f"方向: {signal_details.get('direction')}")
                    logger.info(f"入场价: {signal_details.get('entry_price')}")
                    logger.info(f"止损: {signal_details.get('stop_loss')}")
                    logger.info(f"止盈: {signal_details.get('take_profit')}")
            elif result and result['recommendation'] in ['SKIP', 'WAIT']:
                logger.info("AI建议等待更好的机会")
            elif result is None:
                logger.info("当前没有符合条件的交易机会")

            logger.info(f"=== 分析完成，等待下一个5分钟整点 ===")

        except KeyboardInterrupt:
            logger.info("策略停止")
            break
        except Exception as e:
            logger.error(f"策略运行出错: {e}")
            logger.info("出错后等待1分钟再重试...")
            time.sleep(60)  # 出错后等待1分钟再重试

if __name__ == "__main__":
    main()