# DT-FM 项目环境设置

## 已完成的配置

### 虚拟环境
- 虚拟环境位置: `/home/yufeng.xia/project/DT-FM/env`
- Python 版本: 3.12
- 使用 `fish` shell 时需要特殊激活方式

### 已安装的包

#### Deep Learning 框架
- **PyTorch**: 2.2.0+cu121 (带 CUDA 12.1 支持)
  - `torch.cuda.is_available()` ✓
  - GPU 数量: 8 (NVIDIA RTX A5000)
- **TorchText**: 0.16.2+cpu
- **TorchVision**: 0.17.0+cu121
- **NumPy**: 1.26.4 (PyTorch 兼容版本)

#### CuPy (GPU 计算)
- **CuPy**: 13.6.0-cuda12x
- **注意**: NCCL 支持目前不可用 (见下方问题说明)

#### 数据处理和科学计算
- **SciPy**: 1.17.0
- **Pandas**: 2.3.3
- **Matplotlib**: 3.10.8
- **Six**: 1.17.0 (Python 2/3 兼容性)

### 激活虚拟环境

#### 使用 Fish Shell
```bash
# 方式 1: 直接使用虚拟环境的 Python
/home/yufeng.xia/project/DT-FM/env/bin/python script.py

# 方式 2: 使用虚拟环境的 pip
/home/yufeng.xia/project/DT-FM/env/bin/pip install package_name

# 方式 3: 在 Fish 配置中永久启用 (编辑 ~/.config/fish/config.fish)
set -gx PATH /home/yufeng.xia/project/DT-FM/env/bin $PATH
```

#### 使用 Bash 或 Zsh
```bash
source /home/yufeng.xia/project/DT-FM/env/bin/activate
```

## 已知问题和解决方案

### CuPy NCCL 支持
**问题**: CuPy 编译时未包含 NCCL 支持，导致 `cupy.cuda.nccl` 模块不可用。

**原因**: 预编译的轮子可能没有包含 NCCL，需要从源代码编译 CuPy 并链接本地 NCCL 库。

**解决方案选项**:

1. **安装 NCCL 开发库并从源编译 CuPy** (推荐)
```bash
# 首先安装 NCCL 库
sudo apt-get install libnccl2 libnccl-dev

# 卸载当前 CuPy
/home/yufeng.xia/project/DT-FM/env/bin/pip uninstall cupy-cuda12x -y

# 从源代码编译 (可能需要长时间)
CUPY_INSTALL_USE_CUDA_PATH=1 /home/yufeng.xia/project/DT-FM/env/bin/pip install --no-binary cupy cupy
```

2. **使用 NVIDIA 官方容器镜像** (替代方案)
- 使用 Docker 镜像: `nvidia/cuda:12.1.1-devel-ubuntu22.04`
- 这些镜像预装了 CuPy + NCCL

3. **使用 PyTorch 的分布式后端代替**
- 如果不需要 CuPy/NCCL 特定功能，可修改 `comm/nccl_backend.py` 使用 PyTorch 的 NCCL 后端

## 项目结构

主要模块:
- `dist_runner.py` - 分布式训练脚本
- `foo.py` - 简单的 PyTorch 和 CuPy 加载测试
- `comm/` - 通信模块 (NCCL, 其他)
- `data_parallel/` - 数据并行实现
- `pipeline_parallel/` - 管道并行实现
- `scheduler/` - 调度器
- `task_datasets/` - 数据集加载
- `modules/` - GPT 等模型模块

## 测试虚拟环境

```bash
# 测试 PyTorch
/home/yufeng.xia/project/DT-FM/env/bin/python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 测试 CuPy
/home/yufeng.xia/project/DT-FM/env/bin/python -c "import cupy; print(f'CuPy {cupy.__version__}')"

# 运行简单测试
/home/yufeng.xia/project/DT-FM/env/bin/python /home/yufeng.xia/project/DT-FM/foo.py
```

## 下一步

1. **解决 NCCL 问题** (如果需要分布式训练)
2. **下载数据集** (QQP GLUE 数据集)
3. **配置网络** (设置 GLOO_SOCKET_IFNAME 和 NCCL_SOCKET_IFNAME)
4. **运行训练脚本** (参见 README.md 的 "Manually run cmd on each instance" 部分)

## 参考

- PyTorch 文档: https://pytorch.org/
- CuPy 文档: https://docs.cupy.dev/
- 项目 README: [./README.md](./README.md)
