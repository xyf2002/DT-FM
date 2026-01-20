#!/usr/bin/env fish
# Fish shell activation script for DT-FM project virtual environment

# 添加虚拟环境到 PATH
set -x DT_FM_VENV /home/yufeng.xia/project/DT-FM/env
set -x PATH $DT_FM_VENV/bin $PATH

# 设置提示符前缀
set -x VIRTUAL_ENV $DT_FM_VENV

# 显示激活信息
echo "✓ DT-FM 虚拟环境已激活"
echo "  虚拟环境: $DT_FM_VENV"
echo "  Python: "($DT_FM_VENV/bin/python --version)
echo ""
echo "可用命令:"
echo "  python     - 运行 Python 脚本"
echo "  pip        - 安装/管理包"
echo "  deactivate_dtfm - 停用虚拟环境"
echo ""

# 创建 deactivate 函数
function deactivate_dtfm
    set -e DT_FM_VENV
    set -e VIRTUAL_ENV
    set -x PATH (string match -v $DT_FM_VENV/bin $PATH)
    functions --erase deactivate_dtfm
    echo "✓ DT-FM 虚拟环境已停用"
end
