"""
日志配置模块
支持同时输出到控制台和文件
"""
import logging
import sys
from datetime import datetime
import os


class LoggerConfig:
    """日志配置类"""

    def __init__(self, log_filename='app.log', log_level=logging.INFO):
        self.log_filename = log_filename
        self.log_level = log_level
        self.logger = None

    def setup_logger(self, name=None):
        """
        设置日志器，支持同时输出到控制台和文件

        Args:
            name: 日志器名称，默认使用调用模块的名称

        Returns:
            logging.Logger: 配置好的日志器
        """
        # 创建日志器
        if name:
            self.logger = logging.getLogger(name)
        else:
            # 自动获取调用模块的名称
            import inspect
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get('__name__', 'root')
            self.logger = logging.getLogger(name)

        self.logger.setLevel(self.log_level)

        # 避免重复添加处理器
        if self.logger.handlers:
            return self.logger

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 1. 文件处理器
        file_handler = logging.FileHandler(
            self.log_filename,
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 2. 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 防止日志传播到根日志器
        self.logger.propagate = False

        # 输出初始化信息
        self.logger.info("=" * 60)
        self.logger.info(f"日志系统初始化完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"日志文件: {os.path.abspath(self.log_filename)}")
        self.logger.info(f"日志级别: {logging.getLevelName(self.log_level)}")
        self.logger.info("支持输出到文件和控制台")
        self.logger.info("=" * 60)

        return self.logger

    def get_logger(self):
        """获取配置好的日志器"""
        if self.logger is None:
            self.setup_logger()
        return self.logger

    def set_level(self, level):
        """动态设置日志级别"""
        self.log_level = level
        if self.logger:
            self.logger.setLevel(level)
            for handler in self.logger.handlers:
                handler.setLevel(level)


# 全局日志配置实例
_global_logger_config = None


def get_logger(name=None, log_filename='app.log', log_level=logging.INFO):
    """
    获取日志器的便捷函数

    Args:
        name: 日志器名称
        log_filename: 日志文件名
        log_level: 日志级别

    Returns:
        logging.Logger: 配置好的日志器
    """
    global _global_logger_config

    if _global_logger_config is None:
        _global_logger_config = LoggerConfig(log_filename, log_level)

    return _global_logger_config.setup_logger(name)


def setup_logging(log_filename='app.log', log_level=logging.INFO, name=None):
    """
    设置日志系统的便捷函数

    Args:
        log_filename: 日志文件名
        log_level: 日志级别
        name: 日志器名称

    Returns:
        logging.Logger: 配置好的日志器
    """
    return get_logger(name, log_filename, log_level)


# 预定义的日志级别常量
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def get_log_level_by_name(level_name):
    """根据名称获取日志级别"""
    return LOG_LEVELS.get(level_name.upper(), logging.INFO)