# 使用 Python 官方镜像作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置时区为东八区（Asia/Shanghai）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 修改 APT 下载源为清华源并安装基础的编译工具
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free" > /etc/apt/sources.list \
&& echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free" >> /etc/apt/sources.list \
&& echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list \
&& apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
&& rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt（如果有依赖包的话）
COPY requirements.txt .
COPY .env .

# 安装 Python 依赖
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用文件
COPY scalping_strategy.py .

# 运行 Python 脚本
CMD python -u scalping_strategy.py & sleep 5 && tail -f /app/app.log