# 球形脑空间模拟器 (Spherical Brain Space Simulator)

[English Version](README_en.md) | 中文版

> **原创声明**：本创意由 rubyguyan 于2026年3月首次提出并公开发布。
> 
> 详细创意说明请参阅：[IDEA.md](IDEA.md)

基于 **FlyWire/FlyConnectome** 真实神经元数据的脑空间模拟器。

## 简介

本项目使用果蝇大脑的真实神经元形态数据（SWC格式），在球形三维空间中进行神经元网络模拟。神经元之间通过碰撞自动形成突触连接，模拟真实的神经网络活动。

## 核心创新

- **球形空间约束**：将神经元约束在球形空间内，形成自然的空间梯度
- **真实神经元数据**：使用FlyWire项目的真实神经元形态
- **碰撞驱动连接**：神经元通过物理碰撞自动形成突触
- **营养供给系统**：球心营养充足，边缘稀少
- **神经元生命周期**：分裂、消亡、动态平衡

## 数据来源

- **FlyWire** - 果蝇全脑连接组项目
- **FAFB** (Full Adult Fly Brain) - 成年果蝇全脑数据集
- 神经元ID来自 FlyWire 数据库

## 功能特性

- 真实SWC神经元文件加载
- LIF (Leaky Integrate-and-Fire) 神经元模型
- 神经元信号发送/接收接口
- 碰撞自动形成突触连接
- 营养供给系统（空间梯度）
- 2D/3D可视化切换
- 空间大小动态调整
- 时间比例速度控制
- 集群自动识别
- 信号监视面板

## 安装依赖

```bash
pip install matplotlib numpy
```

## 运行方式

```bash
python brain_space.py
```

### 参数选项

```bash
python brain_space.py --count 10        # 神经元数量
python brain_space.py --category 分类名  # 指定神经元分类
python brain_space.py --radius 1000     # 球形空间半径
```

## 快捷键

| 按键 | 功能 |
|------|------|
| 空格 | 暂停/继续 |
| 2/3 | 切换2D/3D模式 |
| [ ] | 缩小/放大空间 |
| - = | 减慢/加快速度 |
| H | 显示帮助窗口 |
| I | 显示/隐藏信号面板 |
| N | 投入新神经元 |
| E | 向神经元发送输入 |
| S | 保存状态 |
| T | 刺激随机神经元 |
| Q | 退出 |

## 文件结构

```
球形脑空间/
├── brain_space.py      # 主程序
├── swc_neuron.py       # SWC神经元模块
├── IDEA.md             # 创意说明文档
├── neurons/            # 神经元SWC文件
│   └── *.swc          # 真实神经元形态数据
├── docs/               # 文档
└── brain_space_data/   # 运行时数据
```

## SWC文件格式

标准SWC神经元形态格式：
- 每行: 点编号 标签 X Y Z 半径 父节点
- 标签: 0=未定义, 1=胞体(soma), 5=分叉点, 6=端点

## 技术细节

### 坐标转换

FAFB坐标（纳米级）转换为球形空间坐标：
- X: 350000-550000 nm
- Y: 80000-220000 nm  
- Z: 20000-240000 nm

### LIF神经元模型参数

- 静息电位: -70 mV
- 阈值电位: -55 mV
- 重置电位: -75 mV
- 不应期: 2 ms

## 引用

如果你在研究或项目中使用了本创意或代码，请引用：

```
球形脑空间模拟器 (Spherical Brain Space Simulator)
作者: rubyguyan
GitHub: https://github.com/rubyguyan/spherical-brain-space
发布日期: 2026年3月
```

## 许可证

- **代码**：MIT License - 自由使用，请注明出处
- **创意说明**：CC BY 4.0 - 转载请注明出处

## 致谢

- [FlyWire](https://flywire.ai/) - 果蝇全脑连接组项目
- [navis](https://github.com/navis-org/navis) - 神经元形态分析工具
