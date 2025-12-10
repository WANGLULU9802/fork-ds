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

load_dotenv()

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

    def fetch_ohlcv(self, limit=200):
        """获取K线数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
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
        检查入场条件（包含做多和做空）

        做多规则：
        1. HL/LL标签出现
        2. 标签后5根K内收盘大于标签K的最高价
        3. 入场K大小 < ATR的两倍
        4. 止损在标签K最低价
        5. 盈亏比0.5:1

        做空规则：
        1. HH/LH标签出现
        2. 标签后5根K内收盘小于标签K的最低价
        3. 入场K大小 < ATR的两倍
        4. 止损在标签K最高价
        5. 盈亏比0.5:1

        Args:
            df: 完整的K线数据
            current_bar_index: 当前K线索引

        Returns:
            list: 入场信号列表
        """
        signals = []

        # 获取最近的标签（最近20根K线内）
        recent_bars = df.iloc[max(0, current_bar_index-20):current_bar_index+1]
        label_bars = recent_bars[recent_bars['label'].notna()]

        for _, label_bar in label_bars.iterrows():
            label = label_bar['label']
            label_index = label_bar.name
            label_high = label_bar['high'] if not pd.isna(label_bar['pivot_high']) else label_bar['label_value']
            label_low = label_bar['low'] if not pd.isna(label_bar['pivot_low']) else label_bar['label_value']

            # 检查做多条件：HL或LL标签 + 向上突破
            if label in ['HL', 'LL']:
                # 检查标签后5根K线内是否有收盘价突破标签最高价
                bars_after_label = df.iloc[label_index+1:min(label_index+6, current_bar_index+1)]

                for i, bar in bars_after_label.iterrows():
                    if bar['close'] > label_high:
                        # 这是潜在的做多入场K
                        entry_bar = bar
                        entry_index = i

                        # 条件3: 入场K的大小要小于ATR的两倍
                        atr_at_entry = df.iloc[entry_index]['atr']
                        entry_bar_range = entry_bar['high'] - entry_bar['low']

                        if not pd.isna(atr_at_entry) and entry_bar_range <= atr_at_entry * 2:
                            # 计算止损和止盈
                            stop_loss = label_low  # 止损在标签K最低价
                            risk = entry_bar['close'] - stop_loss
                            take_profit = entry_bar['close'] + risk * 0.5  # 0.5:1盈亏比

                            signal = {
                                'direction': 'BUY',
                                'label_type': label,
                                'label_index': label_index,
                                'label_high': label_high,
                                'label_low': label_low,
                                'entry_index': entry_index,
                                'entry_price': entry_bar['close'],
                                'entry_time': entry_bar['timestamp'],
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'risk': risk,
                                'reward': risk * 0.5,
                                'atr_at_entry': atr_at_entry,
                                'entry_bar_range': entry_bar_range,
                                'bars_since_label': entry_index - label_index
                            }
                            signals.append(signal)

            # 检查做空条件：HH或LH标签 + 向下突破
            elif label in ['HH', 'LH']:
                # 检查标签后5根K线内是否有收盘价跌破标签最低价
                bars_after_label = df.iloc[label_index+1:min(label_index+6, current_bar_index+1)]

                for i, bar in bars_after_label.iterrows():
                    if bar['close'] < label_low:
                        # 这是潜在的做空入场K
                        entry_bar = bar
                        entry_index = i

                        # 条件3: 入场K的大小要小于ATR的两倍
                        atr_at_entry = df.iloc[entry_index]['atr']
                        entry_bar_range = entry_bar['high'] - entry_bar['low']

                        if not pd.isna(atr_at_entry) and entry_bar_range <= atr_at_entry * 2:
                            # 计算止损和止盈（做空逻辑相反）
                            stop_loss = label_high  # 止损在标签K最高价
                            risk = stop_loss - entry_bar['close']  # 做空风险是止损减去入场价
                            take_profit = entry_bar['close'] - risk * 0.5  # 0.5:1盈亏比

                            signal = {
                                'direction': 'SELL',
                                'label_type': label,
                                'label_index': label_index,
                                'label_high': label_high,
                                'label_low': label_low,
                                'entry_index': entry_index,
                                'entry_price': entry_bar['close'],
                                'entry_time': entry_bar['timestamp'],
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'risk': risk,
                                'reward': risk * 0.5,
                                'atr_at_entry': atr_at_entry,
                                'entry_bar_range': entry_bar_range,
                                'bars_since_label': entry_index - label_index
                            }
                            signals.append(signal)

        return signals

    def analyze_with_ai(self, signal_data, df):
        """
        使用AI分析交易信号

        Args:
            signal_data: 交易信号数据
            df: 完整的K线数据

        Returns:
            dict: AI分析结果
        """
        if not signal_data:
            return None

        signal = signal_data[0]  # 取最新信号

        # 获取最近150根K线数据用于分析
        recent_bars = df.iloc[max(0, signal['entry_index']-149):signal['entry_index']+1]

        # 构建K线文本
        kline_text = f"最近150根{self.timeframe}K线数据：\n"
        for i, (_, bar) in enumerate(recent_bars.iterrows()):
            trend = "阳线" if bar['close'] > bar['open'] else "阴线"
            change = ((bar['close'] - bar['open']) / bar['open']) * 100
            kline_text += f"K{i+1}: {trend} O:{bar['open']:.2f} C:{bar['close']:.2f} H:{bar['high']:.2f} L:{bar['low']:.2f} V:{bar['volume']:.0f} 涨跌:{change:+.2f}%\n"

        # 根据信号方向调整分析重点
        if signal['direction'] == 'BUY':
            signal_type = "做多"
            breakdown_direction = "向上突破"
            key_level = "阻力位"
            system_focus = "寻找反弹机会，关注支撑位和多头动能"
        else:  # SELL
            signal_type = "做空"
            breakdown_direction = "向下突破"
            key_level = "支撑位"
            system_focus = "寻找下跌机会，关注阻力位和空头动能"

        prompt = f"""
你是一个专业的加密货币剥头皮交易员。
请基于以下{self.symbol} {self.timeframe}数据进行分析：
识别给出的K线中各种K线形态、楔形和其它三段式回调、三角形、双顶，双底，双底牛旗/双顶熊旗、楔形顶/底作为第二个顶/底的双顶/底、双底/顶回调、双头肩顶/底、杯柄底、第一次均线缺口/20缺口K线/移动平均线缺口/k线BODY缺口以及其它各种缺口和测量距离、微型通道、宽幅趋势通道和常见的趋势形态、窄幅交易区间、识别动能反转的标志、掌握鼎峰反转/主要趋势反转/楔形和其它三浪推进反转模式/扩张三角形等各种反转模式、能动态解析识别趋势动能和反转动能以及市场多空力量（以上称为各种价格结构），实时提供精准的入场建议

【技术指标数据】
{kline_text}

【{signal_type}交易信号详情】
- 信号方向: {signal['direction']} ({signal_type})
- 标签类型: {signal['label_type']}
- 标签出现时间: {df.iloc[signal['label_index']]['timestamp']}
- 标签高点: ${signal['label_high']:.2f}
- 标签低点: ${signal['label_low']:.2f}
- 入场时间: {signal['entry_time']}
- 入场价格: ${signal['entry_price']:.2f}
- 止损价格: ${signal['stop_loss']:.2f}
- 止盈价格: ${signal['take_profit']:.2f}
- 风险: ${signal['risk']:.2f}
- 回报: ${signal['reward']:.2f}
- 当前ATR: ${signal['atr_at_entry']:.2f}
- 入场K振幅: ${signal['entry_bar_range']:.2f} (ATR的{signal['entry_bar_range']/signal['atr_at_entry']*100:.1f}%)
- 标签后入场间隔: {signal['bars_since_label']}根K线
- 突破方式: {breakdown_direction}

【{signal_type}分析重点】
{signal_type}信号逻辑分析：
1. {signal['label_type']}标签表明市场出现了{signal_type}前的反转结构
2. 当前价格{breakdown_direction}标签关键{key_level}
3. 风险控制设在标签的相反端，盈亏比为0.5:1的剥头皮策略

【分析要求】
1. 评估这个{signal_type}剥头皮信号的可靠性
2. 分析{breakdown_direction}的力度和市场情绪
3. 考虑{key_level}的有效性和后续走势
4. 评估当前波动率和时机选择
5. 判断短期动能是否支持{signal_type}方向
6. 给出是否建议入场的最终判断

【特别提醒】
- 这是剥头皮策略，重点关注短期价格行为
- 盈亏比较低(0.5:1)，需要高胜率来盈利
- 入场时机和突破质量比长期趋势更重要

请用以下JSON格式回复：
{{
    "recommendation": "ENTER|SKIP|WAIT",
    "confidence": "HIGH|MEDIUM|LOW",
    "reason": "详细分析理由（重点说明{signal_type}逻辑）",
    "risk_assessment": "风险评估（{signal_type}特定风险）",
    "market_context": "市场背景分析",
    "breakdown_quality": "突破质量评估",
    "timing_assessment": "入场时机评估"
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
        """运行完整的策略分析"""
        logger.info(f"开始分析 {self.symbol} {self.timeframe} 剥头皮策略")

        # 获取K线数据
        df = self.fetch_ohlcv(200)
        if df is None:
            logger.error("无法获取K线数据")
            return

        logger.info(f"获取到 {len(df)} 根K线数据")

        # 计算转折点
        df = self.calculate_pivots(df)

        # 计算ATR
        df = self.calculate_atr(df)

        # 识别转折点序列并标记
        df, pivots = self.identify_pivot_sequence(df)

        # 检查入场条件
        current_index = len(df) - 1  # 当前最新K线
        signals = self.check_entry_conditions(df, current_index)

        if signals:
            logger.info(f"发现 {len(signals)} 个交易信号")

            # 使用AI分析最新信号
            ai_result = self.analyze_with_ai(signals, df)

            if ai_result:
                logger.info("=== AI分析结果 ===")
                logger.info(f"建议: {ai_result['recommendation']}")
                logger.info(f"信心: {ai_result['confidence']}")
                logger.info(f"理由: {ai_result['reason']}")
                logger.info(f"风险评估: {ai_result['risk_assessment']}")
                logger.info(f"市场背景: {ai_result['market_context']}")

                return {
                    'signal': signals[0],
                    'ai_analysis': ai_result,
                    'recommendation': ai_result['recommendation']
                }
        else:
            logger.info("当前没有符合条件的交易信号")

        return None

def main():
    """主函数"""
    strategy = ScalpingStrategy(
        symbol='SOL/USDT',
        timeframe='5m',
        length=10
    )

    while True:
        try:
            result = strategy.run_analysis()

            if result and result['recommendation'] == 'ENTER':
                logger.info("🚨 AI建议入场！")
                # 这里可以添加实际的交易执行逻辑

            # 每5分钟检查一次
            logger.info("等待5分钟...")
            time.sleep(300)

        except KeyboardInterrupt:
            logger.info("策略停止")
            break
        except Exception as e:
            logger.error(f"策略运行出错: {e}")
            time.sleep(60)  # 出错后等待1分钟再重试

if __name__ == "__main__":
    main()