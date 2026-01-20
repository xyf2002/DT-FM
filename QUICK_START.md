# 快速参考指南 - DT-FM 项目

## 🚀 虚拟环境激活

### Fish Shell (推荐)
```bash
source /home/yufeng.xia/project/DT-FM/activate.fish
```

### Bash/Zsh
```bash
source /home/yufeng.xia/project/DT-FM/env/bin/activate
```

### 直接使用 (无需激活)
```bash
/home/yufeng.xia/project/DT-FM/env/bin/python script.py
```

---

## ✅ 验证安装

### 快速检查
```bash
/home/yufeng.xia/project/DT-FM/env/bin/python foo.py
# 输出: <== Load torch and cupy. ==>
```

### 详细检查
```bash
/home/yufeng.xia/project/DT-FM/env/bin/python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
EOF
```

---

## 📦 已安装的包

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.2.0+cu121 | 深度学习框架 |
| torchtext | 0.16.2+cpu | 文本处理 |
| torchvision | 0.17.0+cu121 | 计算机视觉 |
| numpy | 1.26.4 | 数值计算 |
| cupy-cuda12x | 13.6.0 | GPU 数组计算 |
| scipy | 1.17.0 | 科学计算 |
| pandas | 2.3.3 | 数据处理 |
| matplotlib | 3.10.8 | 数据可视化 |

---

## 🔧 运行项目

### 单 GPU 训练
```bash
/home/yufeng.xia/project/DT-FM/env/bin/python dist_runner.py \
  --use-cuda \
  --cuda-id 0 \
  --world-size 1 \
  --rank 0 \
  --dist-url tcp://127.0.0.1:9000
```

### 多 GPU 训练 (本地)
```bash
# 在多个终端中分别运行:
/home/yufeng.xia/project/DT-FM/env/bin/python dist_runner.py \
  --use-cuda \
  --cuda-id 0 \
  --world-size 2 \
  --rank 0 \
  --dist-url tcp://127.0.0.1:9000

/home/yufeng.xia/project/DT-FM/env/bin/python dist_runner.py \
  --use-cuda \
  --cuda-id 1 \
  --world-size 2 \
  --rank 1 \
  --dist-url tcp://127.0.0.1:9000
```

---

## ⚠️ 已知问题

### CuPy NCCL 支持缺失
- **症状**: `AttributeError: module 'cupy.cuda' has no attribute 'nccl'`
- **影响**: 分布式训练中的 NCCL 通信不可用
- **解决方案**:
  1. 安装 NCCL 库: `sudo apt-get install libnccl2 libnccl-dev`
  2. 从源编译 CuPy (需要较长时间)
  3. 或使用 PyTorch 的分布式后端

---

## 📚 更多信息

- [SETUP_SUMMARY.txt](./SETUP_SUMMARY.txt) - 安装总结
- [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md) - 详细环境说明
- [README.md](./README.md) - 项目主文档

---

## 🆘 故障排除

### 问题: 找不到 Python 模块
```bash
# 检查 Python 路径
/home/yufeng.xia/project/DT-FM/env/bin/python -c "import sys; print(sys.path)"

# 检查虚拟环境
/home/yufeng.xia/project/DT-FM/env/bin/python -c "import sys; print(sys.prefix)"
```

### 问题: GPU 不可用
```bash
# 检查 CUDA
/home/yufeng.xia/project/DT-FM/env/bin/python -c "import torch; print(torch.cuda.is_available())"

# 检查 GPU 设备
nvidia-smi

# 检查 CUDA 环境变量
echo $CUDA_VISIBLE_DEVICES
```

### 问题: 导入错误
```bash
# 重新安装单个包
/home/yufeng.xia/project/DT-FM/env/bin/pip install --upgrade --force-reinstall torch

# 清理缓存
/home/yufeng.xia/project/DT-FM/env/bin/pip cache purge
```

---

## 💡 提示

1. **虚拟环境路径**: `/home/yufeng.xia/project/DT-FM/env`
2. **Python 可执行路径**: `/home/yufeng.xia/project/DT-FM/env/bin/python`
3. **Pip 可执行路径**: `/home/yufeng.xia/project/DT-FM/env/bin/pip`
4. **项目根目录**: `/home/yufeng.xia/project/DT-FM`

---

最后更新: 2026-01-20
