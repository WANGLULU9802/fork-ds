#!/bin/bash
# Docker 镜像和容器配置
IMAGE_NAME="ema-strategy"
CONTAINER_NAME="ema-strategy-container"
IMAGE_TAG="latest"
# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== EMA Strategy Docker 部署脚本 ===${NC}"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装,请先安装 Docker${NC}"
    exit 1
fi

# 创建日志目录
LOG_DIR="$(pwd)/logs"
if [ ! -d "$LOG_DIR" ]; then
    echo -e "${YELLOW}创建日志目录: ${LOG_DIR}${NC}"
    mkdir -p "$LOG_DIR"
fi

# 停止并删除已存在的容器
echo -e "${YELLOW}检查并清理旧容器...${NC}"
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "停止旧容器..."
    docker stop ${CONTAINER_NAME} 2>/dev/null
    echo "删除旧容器..."
    docker rm ${CONTAINER_NAME} 2>/dev/null
fi

# 构建 Docker 镜像
echo -e "${GREEN}开始构建 Docker 镜像...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# 检查构建是否成功
if [ $? -ne 0 ]; then
    echo -e "${RED}Docker 镜像构建失败！${NC}"
    exit 1
fi

echo -e "${GREEN}Docker 镜像构建成功！${NC}"

# 运行 Docker 容器
echo -e "${GREEN}启动 Docker 容器...${NC}"
docker run -d \
    --name ${CONTAINER_NAME} \
    --restart unless-stopped \
    -v ${LOG_DIR}:/app/logs \
    ${IMAGE_NAME}:${IMAGE_TAG}

# 检查容器是否启动成功
if [ $? -eq 0 ]; then
    echo -e "${GREEN}容器启动成功！${NC}"
    echo -e "${YELLOW}容器名称: ${CONTAINER_NAME}${NC}"
    echo -e "${YELLOW}日志目录: ${LOG_DIR}${NC}"
    echo ""
    echo "常用命令:"
    echo "  查看日志文件: tail -f ${LOG_DIR}/app.log"
    echo "  查看容器日志: docker logs -f ${CONTAINER_NAME}"
    echo "  停止容器: docker stop ${CONTAINER_NAME}"
    echo "  删除容器: docker rm ${CONTAINER_NAME}"
    echo "  进入容器: docker exec -it ${CONTAINER_NAME} /bin/bash"
else
    echo -e "${RED}容器启动失败！${NC}"
    exit 1
fi