#!/bin/bash
# 便捷的开发工作流脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Morphling Development Workflow ===${NC}"

# 函数：显示帮助
show_help() {
    echo "Usage: $0 [COMMAND] [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  build          - 重新构建 Docker 镜像 (GPU 模式)"
    echo "  build-cpu      - 重新构建 Docker 镜像 (CPU 模式)"
    echo "  start          - 启动开发容器 (GPU 模式)"
    echo "  start-cpu      - 启动开发容器 (CPU 模式)"
    echo "  stop           - 停止容器"
    echo "  restart        - 重启容器 (GPU 模式)"
    echo "  restart-cpu    - 重启容器 (CPU 模式)"
    echo "  shell          - 进入容器 shell"
    echo "  rebuild        - 在容器内重新编译 morphling"
    echo "  test           - 运行测试"
    echo "  run [cmd]      - 在容器内运行命令"
    echo "  logs           - 查看容器日志"
    echo "  clean          - 清理容器和卷"
    echo ""
    echo "Examples:"
    echo "  $0 start          # 启动GPU模式"
    echo "  $0 start-cpu      # 启动CPU模式"
    echo "  $0 shell"
    echo "  $0 rebuild"
    echo "  $0 run python3 scripts/run_devices.py --num_devices 4 --model_name facebook/opt-125m --backend proxy"
}

# 检查 docker-compose 是否可用
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: docker-compose is not installed${NC}"
        exit 1
    fi
}

# 函数：构建镜像
build_image() {
    echo -e "${YELLOW}Building Docker image (GPU mode)...${NC}"
    docker-compose build device-emulator
    echo -e "${GREEN}Build completed${NC}"
}

# 函数：构建镜像 (CPU模式)
build_image_cpu() {
    echo -e "${YELLOW}Building Docker image (CPU mode)...${NC}"
    docker-compose -f docker-compose.yml -f docker-compose.cpu.yml build device-emulator
    echo -e "${GREEN}Build completed${NC}"
}

# 函数：启动容器
start_container() {
    echo -e "${YELLOW}Starting development container (GPU mode)...${NC}"
    docker-compose up -d device-emulator
    echo -e "${GREEN}Container started${NC}"
    echo -e "${BLUE}Run '$0 shell' to enter the container${NC}"
}

# 函数：启动容器 (CPU模式)
start_container_cpu() {
    echo -e "${YELLOW}Starting development container (CPU mode)...${NC}"
    docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d device-emulator
    echo -e "${GREEN}Container started${NC}"
    echo -e "${BLUE}Run '$0 shell' to enter the container${NC}"
}

# 函数：停止容器
stop_container() {
    echo -e "${YELLOW}Stopping containers...${NC}"
    docker-compose stop
    echo -e "${GREEN}Containers stopped${NC}"
}

# 函数：重启容器
restart_container() {
    echo -e "${YELLOW}Restarting containers (GPU mode)...${NC}"
    docker-compose restart device-emulator
    echo -e "${GREEN}Container restarted${NC}"
}

# 函数：重启容器 (CPU模式)
restart_container_cpu() {
    echo -e "${YELLOW}Restarting containers (CPU mode)...${NC}"
    docker-compose -f docker-compose.yml -f docker-compose.cpu.yml restart device-emulator
    echo -e "${GREEN}Container restarted${NC}"
}

# 函数：进入容器shell
enter_shell() {
    echo -e "${YELLOW}Entering container shell...${NC}"
    docker-compose exec device-emulator bash
}

# 函数：在容器内重新编译
rebuild_in_container() {
    echo -e "${YELLOW}Rebuilding morphling in container...${NC}"
    docker-compose exec device-emulator bash /app/scripts/dev_build.sh
    echo -e "${GREEN}Rebuild completed${NC}"
}

# 函数：运行命令
run_command() {
    shift  # 移除 'run' 参数
    echo -e "${YELLOW}Running command in container: $@${NC}"
    docker-compose exec device-emulator bash -c "$*"
}

# 函数：查看日志
show_logs() {
    echo -e "${YELLOW}Showing container logs...${NC}"
    docker-compose logs -f device-emulator
}

# 函数：清理
clean_up() {
    echo -e "${YELLOW}Cleaning up containers and volumes...${NC}"
    docker-compose down -v
    docker-compose down --rmi local
    echo -e "${GREEN}Cleanup completed${NC}"
}

# 函数：运行测试
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    docker-compose exec device-emulator bash -c "cd /app && python3 -m pytest tests/ -v"
}

# 主逻辑
check_docker_compose

case "${1:-help}" in
    "build")
        build_image
        ;;
    "build-cpu")
        build_image_cpu
        ;;
    "start")
        start_container
        ;;
    "start-cpu")
        start_container_cpu
        ;;
    "stop")
        stop_container
        ;;
    "restart")
        restart_container
        ;;
    "restart-cpu")
        restart_container_cpu
        ;;
    "shell")
        enter_shell
        ;;
    "rebuild")
        rebuild_in_container
        ;;
    "test")
        run_tests
        ;;
    "run")
        run_command "$@"
        ;;
    "logs")
        show_logs
        ;;
    "clean")
        clean_up
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
