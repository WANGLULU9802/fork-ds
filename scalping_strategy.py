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
        入场条件检查 - 先用代码判断价格突破，再收集数据给AI进行形态分析

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

        # 代码实现：检查价格突破条件
        price_breakthrough = self.check_price_breakthrough(df, current_bar_index)

        # 准备给AI的数据
        data_for_ai = {
            'df': recent_bars,
            'labels': all_labels,
            'recent_6_labels': recent_6_labels,
            'current_index': current_bar_index,
            'has_labels': len(recent_6_labels) > 0,  # 只检查最近6根K线是否有标签
            'price_breakthrough': price_breakthrough  # 添加价格突破判断结果
        }

        return data_for_ai

    def check_price_breakthrough(self, df, current_bar_index):
        """
        用代码判断价格是否突破标签高低点

        Args:
            df: 完整的K线数据
            current_bar_index: 当前K线索引

        Returns:
            dict: 价格突破判断结果
        """
        # 获取最近6根K线内的标签
        start_idx = max(0, current_bar_index - 5)
        recent_6_bars = df.iloc[start_idx:current_bar_index+1]
        recent_6_labels = recent_6_bars[recent_6_bars['label'].notna()]

        if recent_6_labels.empty:
            return {
                'has_breakthrough': False,
                'direction': None,
                'label_info': None,
                'entry_bar_info': None
            }

        # 检查每个标签后的5根K线是否有突破
        for _, label_bar in recent_6_labels.iterrows():
            label_idx = label_bar.name
            label_type = label_bar['label']

            # 确定检查范围（标签后5根K线）
            check_start = label_idx + 1
            check_end = min(label_idx + 6, current_bar_index + 1)

            if check_start >= len(df):
                continue

            # 检查范围内的每根K线
            for entry_idx in range(check_start, check_end):
                entry_bar = df.iloc[entry_idx]

                # 计算ATR（用于K线大小过滤）
                atr_period = 14
                atr_start = max(0, entry_idx - atr_period + 1)
                atr_data = df.iloc[atr_start:entry_idx+1]
                if len(atr_data) < atr_period:
                    continue

                # 计算ATR
                atr = self.calculate_simple_atr(atr_data)

                # 计算K线实体大小
                bar_size = abs(entry_bar['close'] - entry_bar['open'])

                # 检查K线大小是否超过ATR两倍
                if bar_size > 2 * atr:
                    continue

                # 检查引线大小
                upper_shadow = entry_bar['high'] - max(entry_bar['open'], entry_bar['close'])
                lower_shadow = min(entry_bar['open'], entry_bar['close']) - entry_bar['low']

                # 做多条件：HL或LL标签，收盘价突破标签最高点，无长上引线
                if label_type in ['HL', 'LL']:
                    if entry_bar['close'] > label_bar['label_value']:
                        # 检查上引线是否过大（上引线不超过实体的50%）
                        body_size = abs(entry_bar['close'] - entry_bar['open'])
                        if upper_shadow <= body_size * 0.5:
                            return {
                                'has_breakthrough': True,
                                'direction': 'BUY',
                                'label_info': {
                                    'label_type': label_type,
                                    'label_price': label_bar['label_value'],
                                    'label_index': label_idx,
                                    'stop_loss': label_bar['low']  # 止损设在标签K最低点
                                },
                                'entry_bar_info': {
                                    'entry_price': entry_bar['close'],
                                    'entry_index': entry_idx,
                                    'bar_size': bar_size,
                                    'atr': atr,
                                    'upper_shadow': upper_shadow,
                                    'lower_shadow': lower_shadow
                                }
                            }

                # 做空条件：HH或LH标签，收盘价跌破标签最低点，无长下引线
                elif label_type in ['HH', 'LH']:
                    if entry_bar['close'] < label_bar['label_value']:
                        # 检查下引线是否过大（下引线不超过实体的50%）
                        body_size = abs(entry_bar['close'] - entry_bar['open'])
                        if lower_shadow <= body_size * 0.5:
                            return {
                                'has_breakthrough': True,
                                'direction': 'SELL',
                                'label_info': {
                                    'label_type': label_type,
                                    'label_price': label_bar['label_value'],
                                    'label_index': label_idx,
                                    'stop_loss': label_bar['high']  # 止损设在标签K最高点
                                },
                                'entry_bar_info': {
                                    'entry_price': entry_bar['close'],
                                    'entry_index': entry_idx,
                                    'bar_size': bar_size,
                                    'atr': atr,
                                    'upper_shadow': upper_shadow,
                                    'lower_shadow': lower_shadow
                                }
                            }

        return {
            'has_breakthrough': False,
            'direction': None,
            'label_info': None,
            'entry_bar_info': None
        }

    def calculate_simple_atr(self, df, period=14):
        """计算简单的ATR"""
        if len(df) < period:
            return 0

        df = df.copy()
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))

        df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        return df['tr'].mean()

    def analyze_with_ai(self, data_for_ai, df):
        """
        使用AI分析K线形态（价格突破已由代码判断）

        Args:
            data_for_ai: 包含K线数据和标签信息的数据
            df: 完整的K线数据

        Returns:
            dict: AI分析结果
        """
        # 检查是否有价格突破
        if not data_for_ai['price_breakthrough']['has_breakthrough']:
            logger.info("代码检查：没有符合条件的价格突破，不进行AI分析")
            return None

        # 检查最后一根K线是否有标签，如果有则不进行分析
        recent_bars = data_for_ai['df']
        last_bar = recent_bars.iloc[-1]
        if not pd.isna(last_bar['label']):
            logger.info(f"最后一根K线有标签 {last_bar['label']}，不进行AI分析，等待后续K线")
            return None

        # 获取价格突破信息
        breakthrough = data_for_ai['price_breakthrough']
        direction = breakthrough['direction']
        label_info = breakthrough['label_info']
        entry_info = breakthrough['entry_bar_info']

        # 构建K线文本，包含技术指标
        kline_text = f"最近30根{self.timeframe}K线数据及指标：\n"
        for i, (_, bar) in enumerate(recent_bars.iterrows()):
            change = ((bar['close'] - bar['open']) / bar['open']) * 100

            # 获取ATR值
            atr = bar['atr'] if 'atr' in bar and not pd.isna(bar['atr']) else 0

            # 获取EMA20值
            ema20 = bar['close'] * 0.9 if 'ema20' not in bar else bar['ema20']

            # 检查是否有标签
            label_info_text = ""
            if not pd.isna(bar['label']):
                label_info_text = f" 标签:{bar['label']}"

            # 标记入场K线
            entry_mark = ""
            if entry_info and i == recent_bars.index.get_loc(entry_info['entry_index']):
                entry_mark = " [入场K]"

            kline_text += f"K{i+1}: O:{bar['open']:.2f} C:{bar['close']:.2f} H:{bar['high']:.2f} L:{bar['low']:.2f} V:{bar['volume']:.0f} 涨跌:{change:+.2f}% EMA20:{ema20:.2f} ATR:{atr:.4f}{label_info_text}{entry_mark}\n"

        # 构建标签信息
        labels = data_for_ai['labels']
        label_text = "发现的标签信息：\n"
        for idx, label_bar in labels.iterrows():
            # 计算这是第多少根K线（从最近30根的开始算起）
            k_index = 30 - len(recent_bars) + recent_bars.index.get_loc(idx) + 1
            label_text += f"- K{k_index}: 标签 {label_bar['label']} 价格: {label_bar['label_value']:.2f}\n"

        prompt = f"""
你是一个专业的加密货币剥头皮交易员，专注于K线形态分析。

【市场数据】
交易对: {self.symbol}
时间周期: {self.timeframe}

【代码已确认的价格突破信息】
方向: {direction}
标签类型: {label_info['label_type']}
标签价格: {label_info['label_price']:.2f}
入场价格: {entry_info['entry_price']:.2f}
止损位: {label_info['stop_loss']:.2f}
K线大小: {entry_info['bar_size']:.4f}
ATR: {entry_info['atr']:.4f}

{kline_text}

{label_text}

【分析任务】
代码已经确认价格突破条件满足，现在需要你分析K线形态来确认交易信号的可靠性。

请识别并分析以下形态：
1. K线形态（锤子线、十字星、吞没形态、流星线等）
2. 反转形态（双顶/底、头肩形、楔形、V形反转等）
3. 持续形态（三角形、旗形、矩形等）
4. 缺口和测量距离（突破缺口、衰竭缺口等）
5. 趋势通道和交易区间（上升趋势线、下降趋势线、支撑阻力位等）
6. 动能反转信号（背离、超买超卖反转等）

【分析要点】
1. 入场K线的形态是否支持突破方向？
2. 突破前的K线组合是否形成反转或持续形态？
3. 是否存在确认信号（如多个看涨/看跌形态组合）？
4. 当前价格位置是否处于关键的技术位？
5. 成交量是否支持突破的有效性？

【风险评估】
1. 形态的可靠性程度
2. 假突破的可能性
3. 市场整体趋势的方向性
4. 潜在的风险因素

请基于形态分析判断是否应该入场，用以下JSON格式回复：
{{
    "recommendation": "BUY|SELL|SKIP|WAIT",
    "confidence": "HIGH|MEDIUM|LOW",
    "reason": "基于K线形态的详细分析理由",
    "pattern_analysis": {{
        "entry_bar_pattern": "入场K线的具体形态",
        "preceding_patterns": "突破前的关键形态组合",
        "confirmation_signals": "确认信号列表",
        "reversal_or_continuation": "反转或持续形态判断",
        "volume_analysis": "成交量分析"
    }},
    "signal_details": {{
        "direction": "{direction}",
        "entry_price": {entry_info['entry_price']},
        "stop_loss": {label_info['stop_loss']},
        "take_profit": "止盈位（0.5:1盈亏比）",
        "pattern_strength": "形态强度评估"
    }},
    "risk_assessment": "基于形态的风险评估",
    "market_context": "市场背景和整体趋势分析",
    "overall_signal_quality": "综合信号质量评分（1-10）"
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
        df = self.fetch_ohlcv(50)
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
            logger.info("=== AI形态分析结果 ===")
            logger.info(f"分析时间（东八区）: {current_time}")
            logger.info(f"建议: {ai_result['recommendation']}")
            logger.info(f"信心: {ai_result['confidence']}")
            logger.info(f"理由: {ai_result['reason']}")

            # 输出形态分析
            if 'pattern_analysis' in ai_result:
                pattern = ai_result['pattern_analysis']
                logger.info(f"入场K线形态: {pattern.get('entry_bar_pattern', 'N/A')}")
                logger.info(f"突破前形态: {pattern.get('preceding_patterns', 'N/A')}")
                logger.info(f"形态强度: {ai_result['signal_details'].get('pattern_strength', 'N/A')}")
                logger.info(f"信号质量评分: {ai_result.get('overall_signal_quality', 'N/A')}/10")

            # 输出价格突破信息（由代码判断）
            breakthrough = data_for_ai['price_breakthrough']
            if breakthrough['has_breakthrough']:
                logger.info("=== 代码确认的价格突破 ===")
                logger.info(f"突破方向: {breakthrough['direction']}")
                logger.info(f"标签: {breakthrough['label_info']['label_type']} @ {breakthrough['label_info']['label_price']:.2f}")
                logger.info(f"入场价: {breakthrough['entry_bar_info']['entry_price']:.2f}")
                logger.info(f"止损: {breakthrough['label_info']['stop_loss']:.2f}")

            return {
                'data_for_ai': data_for_ai,
                'ai_analysis': ai_result,
                'recommendation': ai_result['recommendation'],
                'price_breakthrough': breakthrough
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
    logger.info("策略已更新：价格突破由代码判断，AI专注形态分析")

    while True:
        try:
            # 等待到下一个5分钟整点
            wait_until_next_5min()

            # 在5分钟整点执行分析
            current_time = get_beijing_time().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"=== 开始5分钟整点分析（{current_time}） ===")

            result = strategy.run_analysis()

            if result:
                # 有价格突破且AI分析完成
                if result['recommendation'] in ['BUY', 'SELL']:
                    logger.info("🚨 代码确认突破 + AI形态确认！建议入场！")
                    breakthrough = result['price_breakthrough']
                    signal_details = result['ai_analysis'].get('signal_details', {})

                    # 输出交易详情
                    logger.info(f"交易方向: {breakthrough['direction']}")
                    logger.info(f"入场价格: {breakthrough['entry_bar_info']['entry_price']:.2f}")
                    logger.info(f"止损价格: {breakthrough['label_info']['stop_loss']:.2f}")

                    # 计算止盈位（0.5:1盈亏比）
                    risk = abs(breakthrough['entry_bar_info']['entry_price'] - breakthrough['label_info']['stop_loss'])
                    if breakthrough['direction'] == 'BUY':
                        take_profit = breakthrough['entry_bar_info']['entry_price'] + risk * 0.5
                    else:
                        take_profit = breakthrough['entry_bar_info']['entry_price'] - risk * 0.5
                    logger.info(f"止盈价格: {take_profit:.2f}")

                    # 这里可以添加实际的交易执行逻辑

                elif result['recommendation'] in ['SKIP', 'WAIT']:
                    logger.info("AI基于形态分析建议等待更好的机会")
            else:
                # 没有价格突破或没有AI分析
                logger.info("当前没有符合条件的交易机会（价格突破条件未满足）")

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