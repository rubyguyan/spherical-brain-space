#!/usr/bin/env python3
"""
球形脑空间模拟器 - 真实神经元版本 v0.0.1
=====================================
功能：
├── 真实SWC神经元文件加载
├── LIF神经元模型（真实电生理信号）
├── 神经元信号发送/接收接口
├── 碰撞自动形成突触连接
├── 营养供给系统（空间梯度）
├── 2D/3D可视化切换
├── 空间大小动态调整
├── 时间比例速度控制
├── 集群自动识别
├── 信息传递系统
└── 信号查看窗口

运行方式：
    python brain_space.py [--real] [--category 分类名] [--count 数量]

参数：
    --real      使用真实SWC神经元（默认使用模拟神经元）
    --category  指定神经元分类目录
    --count     神经元数量（默认10个）

快捷键：
    空格: 暂停/继续
    2:   切换到2D模式
    3:   切换到3D模式
    [:   缩小空间 (半径-10%)
    ]:   放大空间 (半径+10%)
    -:   减慢速度 (时间比例降低)
    =:   加快速度 (时间比例提高)
    H:   显示帮助窗口
    I:   显示信号窗口
    T:   刺激随机神经元（发送信号）
    S:   保存状态
    Q:   退出
"""

import random
import math
import json
import os
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from collections import deque

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets
    import numpy as np
    
    matplotlib.rcParams['figure.raise_window'] = False
    
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将使用纯文本模式")
    print("安装命令: pip install matplotlib numpy")

try:
    from swc_neuron import (
        SWCLoader, LIFNeuron, NeuronNetwork, 
        NeuronMorphology
    )
    HAS_SWC_MODULE = True
except ImportError:
    HAS_SWC_MODULE = False
    print("警告: swc_neuron模块未找到，将使用模拟神经元")


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self) -> 'Vector3':
        m = self.magnitude()
        if m == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x/m, self.y/m, self.z/m)
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __neg__(self) -> 'Vector3':
        return Vector3(-self.x, -self.y, -self.z)
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    @staticmethod
    def from_list(lst: List[float]) -> 'Vector3':
        return Vector3(lst[0], lst[1], lst[2])


@dataclass
class Signal:
    signal_type: str
    source_id: str
    target_id: Optional[str]
    content: str
    step: int
    signal_strength: float = 1.0
    detail: str = ""
    
    def to_dict(self) -> dict:
        return {
            'type': self.signal_type,
            'source': self.source_id,
            'target': self.target_id,
            'content': self.content,
            'step': self.step,
            'strength': self.signal_strength,
            'detail': self.detail
        }
    
    def __str__(self) -> str:
        if self.target_id:
            return f"[步骤{self.step}] {self.source_id} → {self.target_id}: {self.content}"
        return f"[步骤{self.step}] {self.source_id} → 外部: {self.content}"


@dataclass
class InputRecord:
    content: str
    target_neuron: str
    step: int
    result: str = ""
    
    def to_dict(self) -> dict:
        return {'content': self.content, 'target': self.target_neuron, 
                'step': self.step, 'result': self.result}


@dataclass
class OutputRecord:
    source_neuron: str
    content: str
    step: int
    trigger: str = ""
    
    def to_dict(self) -> dict:
        return {'source': self.source_neuron, 'content': self.content,
                'step': self.step, 'trigger': self.trigger}


class InputOutputManager:
    def __init__(self, max_history: int = 500):
        self.input_records: deque = deque(maxlen=max_history)
        self.output_records: deque = deque(maxlen=max_history)
        self.neuron_outputs: Dict[str, List[OutputRecord]] = {}
    
    def send_input(self, content: str, neuron_id: str, step: int) -> InputRecord:
        record = InputRecord(content=content, target_neuron=neuron_id, step=step)
        self.input_records.append(record)
        print(f"📥 输入发送: \"{content}\" → {neuron_id}")
        return record
    
    def record_output(self, neuron_id: str, content: str, step: int, trigger: str = "") -> OutputRecord:
        record = OutputRecord(source_neuron=neuron_id, content=content, 
                              step=step, trigger=trigger)
        self.output_records.append(record)
        
        if neuron_id not in self.neuron_outputs:
            self.neuron_outputs[neuron_id] = []
        self.neuron_outputs[neuron_id].append(record)
        
        return record
    
    def get_recent_inputs(self, count: int = 20) -> List[InputRecord]:
        return list(self.input_records)[-count:]
    
    def get_recent_outputs(self, count: int = 20) -> List[OutputRecord]:
        return list(self.output_records)[-count:]
    
    def get_stats(self) -> dict:
        return {
            'total_inputs': len(self.input_records),
            'total_outputs': len(self.output_records),
            'neurons_with_outputs': len(self.neuron_outputs)
        }


class SignalManager:
    SIGNAL_TYPES = ['脉冲', '振荡', '持续', '爆发', '静默']
    SIGNAL_CONTENTS = ['激活', '抑制', '调节', '同步', '反馈', '请求', '响应', '状态']
    DETAILED_MESSAGES = {
        '激活': ['请求目标神经元提高活跃度', '发送兴奋性信号', '触发下游反应'],
        '抑制': ['降低目标神经元活跃度', '发送抑制性信号', '阻止过度兴奋'],
        '调节': ['调整突触权重', '优化信号传递效率', '平衡网络活动'],
        '同步': ['协调集群节律', '统一发放频率', '建立振荡模式'],
        '反馈': ['报告当前状态', '确认信号接收', '请求进一步指令'],
        '请求': ['请求资源分配', '询问目标状态', '建立新连接请求'],
        '响应': ['响应之前请求', '提供状态信息', '确认连接建立'],
        '状态': ['广播健康状态', '报告饱腹程度', '更新位置信息']
    }
    
    def __init__(self, max_history: int = 500):
        self.signals: deque = deque(maxlen=max_history)
        self.neuron_signals: Dict[str, List[Signal]] = {}
        self.cluster_signals: Dict[str, List[Signal]] = {}
    
    def generate_signal(self, source_id: str, target_id: Optional[str], 
                        step: int, is_cluster: bool = False) -> Signal:
        signal_type = random.choice(self.SIGNAL_TYPES)
        content = random.choice(self.SIGNAL_CONTENTS)
        strength = random.uniform(0.5, 1.5)
        
        detail = random.choice(self.DETAILED_MESSAGES.get(content, ['传递信息']))
        
        signal = Signal(
            signal_type=signal_type,
            source_id=source_id,
            target_id=target_id,
            content=f"{signal_type}:{content}",
            step=step,
            signal_strength=strength,
            detail=detail
        )
        
        self.signals.append(signal)
        
        if source_id not in self.neuron_signals:
            self.neuron_signals[source_id] = []
        self.neuron_signals[source_id].append(signal)
        
        if is_cluster:
            if source_id not in self.cluster_signals:
                self.cluster_signals[source_id] = []
            self.cluster_signals[source_id].append(signal)
        
        return signal
    
    def get_recent_signals(self, count: int = 50) -> List[Signal]:
        return list(self.signals)[-count:]
    
    def get_neuron_signals(self, neuron_id: str, count: int = 20) -> List[Signal]:
        signals = self.neuron_signals.get(neuron_id, [])
        return signals[-count:]
    
    def get_cluster_signals(self, cluster_id: str, count: int = 20) -> List[Signal]:
        signals = self.cluster_signals.get(cluster_id, [])
        return signals[-count:]
    
    def get_stats(self) -> dict:
        return {
            'total_signals': len(self.signals),
            'neurons_with_signals': len(self.neuron_signals),
            'clusters_with_signals': len(self.cluster_signals)
        }


@dataclass
class Neuron:
    id: str
    name: str
    position: Vector3
    satiety: float = 0.5
    synapses: List[str] = field(default_factory=list)
    age: int = 0
    division_count: int = 0
    steps_without_connection: int = 0
    is_division_child: bool = False
    parent_id: str = ""
    signal_cooldown: int = 0
    is_real_neuron: bool = False
    lif_neuron: Optional['LIFNeuron'] = None
    morphology: Optional['NeuronMorphology'] = None
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'position': self.position.to_list(),
            'satiety': self.satiety,
            'synapses': self.synapses,
            'age': self.age,
            'division_count': self.division_count,
            'steps_without_connection': self.steps_without_connection,
            'is_division_child': self.is_division_child,
            'parent_id': self.parent_id,
            'is_real_neuron': self.is_real_neuron
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'Neuron':
        return Neuron(
            id=d['id'],
            name=d['name'],
            position=Vector3.from_list(d['position']),
            satiety=d['satiety'],
            synapses=d['synapses'],
            age=d['age'],
            division_count=d['division_count'],
            steps_without_connection=d['steps_without_connection'],
            is_division_child=d.get('is_division_child', False),
            parent_id=d.get('parent_id', ""),
            is_real_neuron=d.get('is_real_neuron', False)
        )
    
    def get_membrane_potential(self) -> float:
        if self.is_real_neuron and self.lif_neuron:
            return self.lif_neuron.membrane_potential
        return -70.0 + (self.satiety - 0.5) * 30
    
    def get_firing_rate(self) -> float:
        if self.is_real_neuron and self.lif_neuron:
            return self.lif_neuron.get_firing_rate()
        return 0.0
    
    def stimulate(self, current: float):
        if self.is_real_neuron and self.lif_neuron:
            self.lif_neuron.inject_current(current)
            return True
        return False


class RealNeuronManager:
    def __init__(self):
        self.network: Optional[NeuronNetwork] = None
        self.morphologies: List[NeuronMorphology] = []
        self.neuron_map: Dict[str, Neuron] = {}
        self.use_real_neurons = False
    
    def initialize(self, count: int = 10, category: str = None) -> bool:
        if not HAS_SWC_MODULE:
            print("警告: SWC模块不可用，无法加载真实神经元")
            return False
        
        try:
            self.morphologies = SWCLoader.load_random_neurons(count, category)
            
            if not self.morphologies:
                print("警告: 未能加载任何SWC神经元")
                return False
            
            self.network = NeuronNetwork()
            
            for morph in self.morphologies:
                lif = LIFNeuron(morphology=morph)
                self.network.add_neuron(lif)
            
            self.use_real_neurons = True
            print(f"✓ 加载了 {len(self.morphologies)} 个真实SWC神经元")
            print(f"✓ 神经元文件已复制到: {SWCLoader.TARGET_DIR}")
            return True
            
        except Exception as e:
            print(f"错误: 加载真实神经元失败: {e}")
            return False
    
    def create_neuron_from_morphology(self, morph: NeuronMorphology, R: float) -> Neuron:
        if morph.soma_position:
            x, y, z = SWCLoader.transform_fafb_to_sphere(
                morph.soma_position[0], 
                morph.soma_position[1], 
                morph.soma_position[2],
                R
            )
            position = Vector3(x, y, z)
        else:
            center = morph.get_center()
            x, y, z = SWCLoader.transform_fafb_to_sphere(
                center[0], center[1], center[2], R
            )
            position = Vector3(x, y, z)
        
        neuron = Neuron(
            id=morph.neuron_id or f"neuron_{random.randint(1000, 9999)}",
            name=morph.name[:15] if morph.name else f"N_{morph.neuron_id[:6]}",
            position=position,
            is_real_neuron=True
        )
        
        if self.network and morph.neuron_id in self.network.neurons:
            neuron.lif_neuron = self.network.neurons[morph.neuron_id]
            neuron.morphology = morph
        
        self.neuron_map[neuron.id] = neuron
        return neuron
    
    def step(self, dt: float = 1.0):
        if self.network:
            self.network.step(dt)
    
    def connect_real_neurons(self, source_id: str, target_id: str, weight: float = 1.0):
        if self.network:
            return self.network.connect_neurons(source_id, target_id, weight, True)
        return False
    
    def stimulate_neuron(self, neuron_id: str, current: float):
        if self.network:
            self.network.stimulate_neuron(neuron_id, current)
    
    def get_real_signals(self, count: int = 50) -> List:
        if self.network:
            return self.network.get_recent_signals(count)
        return []
    
    def get_network_stats(self) -> dict:
        if self.network:
            return self.network.get_network_stats()
        return {}


class ClusterManager:
    def __init__(self):
        self.clusters: Dict[str, Set[str]] = {}
        self.neuron_to_cluster: Dict[str, str] = {}
    
    def update(self, neurons: List[Neuron]) -> bool:
        visited = set()
        new_clusters = {}
        neuron_map = {n.id: n for n in neurons}
        
        for neuron in neurons:
            if neuron.id not in visited:
                cluster = self._bfs_cluster(neuron, neuron_map, visited)
                if len(cluster) >= 2:
                    cluster_id = f"集群_{len(new_clusters) + 1}"
                    new_clusters[cluster_id] = cluster
        
        self.clusters = new_clusters
        self.neuron_to_cluster = {}
        for cid, members in new_clusters.items():
            for nid in members:
                self.neuron_to_cluster[nid] = cid
        
        return True
    
    def _bfs_cluster(self, start: Neuron, neuron_map: Dict, visited: Set) -> Set[str]:
        cluster = set()
        queue = [start]
        
        while queue:
            neuron = queue.pop(0)
            if neuron.id in visited:
                continue
            visited.add(neuron.id)
            cluster.add(neuron.id)
            
            for connected_id in neuron.synapses:
                if connected_id not in visited and connected_id in neuron_map:
                    queue.append(neuron_map[connected_id])
        
        return cluster
    
    def get_stats(self, total_neurons: int) -> dict:
        clustered = sum(len(m) for m in self.clusters.values())
        single = total_neurons - clustered
        
        return {
            'total_neurons': total_neurons,
            'single_neurons': single,
            'cluster_count': len(self.clusters),
            'clusters': [
                {'id': cid, 'size': len(members)}
                for cid, members in sorted(self.clusters.items(), key=lambda x: -len(x[1]))
            ]
        }
    
    def get_cluster_bonus(self, neuron_id: str) -> float:
        if neuron_id not in self.neuron_to_cluster:
            return 0.0
        
        cluster_id = self.neuron_to_cluster[neuron_id]
        size = len(self.clusters[cluster_id])
        
        if size == 2:
            return 0.05
        elif size == 3:
            return 0.10
        else:
            return 0.10 + (size - 3) * 0.03
    
    def create_cluster_folder(self, cluster_id: str, neurons: List['Neuron'], base_dir: str) -> str:
        import shutil
        
        cluster_dir = os.path.join(base_dir, "clusters", cluster_id)
        os.makedirs(cluster_dir, exist_ok=True)
        
        if cluster_id not in self.clusters:
            return cluster_dir
        
        member_ids = self.clusters[cluster_id]
        neuron_map = {n.id: n for n in neurons}
        
        info_file = os.path.join(cluster_dir, "cluster_info.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"集群ID: {cluster_id}\n")
            f.write(f"成员数量: {len(member_ids)}\n")
            f.write(f"创建时间: {datetime.now().isoformat()}\n")
            f.write("\n成员列表:\n")
            for nid in member_ids:
                if nid in neuron_map:
                    n = neuron_map[nid]
                    f.write(f"  - {n.name} (ID: {nid})\n")
        
        return cluster_dir
    
    def save_all_clusters(self, neurons: List['Neuron'], base_dir: str) -> int:
        saved = 0
        for cluster_id in self.clusters:
            self.create_cluster_folder(cluster_id, neurons, base_dir)
            saved += 1
        return saved


class SpaceAdjuster:
    MIN_RADIUS = 200
    MAX_RADIUS = 5000
    DEFAULT_RADIUS = 1000
    
    def __init__(self):
        self.R = self.DEFAULT_RADIUS
    
    def shrink(self, factor: float = 0.9) -> float:
        self.R = max(self.MIN_RADIUS, self.R * factor)
        return self.R
    
    def expand(self, factor: float = 1.1) -> float:
        self.R = min(self.MAX_RADIUS, self.R * factor)
        return self.R
    
    def adjust_positions(self, neurons: List[Neuron], old_R: float):
        if old_R == 0 or self.R == old_R:
            return
        ratio = self.R / old_R
        for neuron in neurons:
            neuron.position = neuron.position * ratio


class TimeController:
    SPEED_LEVELS = [
        {'name': '慢速', 'ratio': 120, 'steps_per_frame': 1},
        {'name': '中速', 'ratio': 10, 'steps_per_frame': 5},
        {'name': '快速', 'ratio': 1, 'steps_per_frame': 10},
        {'name': '极速', 'ratio': 0.5, 'steps_per_frame': 20},
        {'name': '超速', 'ratio': 0.1, 'steps_per_frame': 50},
    ]
    
    def __init__(self):
        self.current_level = 1
    
    @property
    def speed_name(self) -> str:
        return self.SPEED_LEVELS[self.current_level]['name']
    
    @property
    def steps_per_frame(self) -> int:
        return self.SPEED_LEVELS[self.current_level]['steps_per_frame']
    
    def slower(self):
        if self.current_level > 0:
            self.current_level -= 1
        return self.speed_name
    
    def faster(self):
        if self.current_level < len(self.SPEED_LEVELS) - 1:
            self.current_level += 1
        return self.speed_name
    
    def get_info(self) -> str:
        level = self.SPEED_LEVELS[self.current_level]
        return f"{level['name']} (培养皿{level['ratio']}秒=外界1秒)"


class BrainSpaceSimulator:
    CONNECTION_DISTANCE = 20
    MAX_SYNAPSES = 3
    MAX_DIVISIONS = 99
    DIVISION_MIN_STEP = 299
    SIGNAL_INTERVAL = 30
    
    def __init__(self, R: int = 1000, initial_count: int = 33, category: str = None):
        self.neurons: List[Neuron] = []
        self.cluster_manager = ClusterManager()
        self.space_adjuster = SpaceAdjuster()
        self.space_adjuster.R = R
        self.time_controller = TimeController()
        self.signal_manager = SignalManager()
        self.real_neuron_manager = RealNeuronManager()
        self.io_manager = InputOutputManager()
        
        self.current_step = 0
        self.paused = False
        self.use_real_neurons = True
        
        if HAS_SWC_MODULE:
            success = self._init_real_neurons(initial_count, category)
            if not success:
                raise RuntimeError("无法加载真实神经元，请检查SWC文件路径")
        else:
            raise RuntimeError("需要swc_neuron模块才能运行")
        
        self.data_dir = "brain_space_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    @property
    def R(self) -> float:
        return self.space_adjuster.R
    
    def _init_real_neurons(self, count: int, category: str = None) -> bool:
        if not self.real_neuron_manager.initialize(count, category):
            return False
        
        for morph in self.real_neuron_manager.morphologies:
            neuron = self.real_neuron_manager.create_neuron_from_morphology(morph, self.R)
            self.neurons.append(neuron)
        
        print(f"✓ 初始化 {len(self.neurons)} 个真实SWC神经元")
        return True
    
    def stimulate_neuron(self, neuron_id: str = None, current: float = 5.0):
        if neuron_id is None:
            neuron = random.choice(self.neurons)
            neuron_id = neuron.id
        else:
            neuron = next((n for n in self.neurons if n.id == neuron_id), None)
        
        if neuron:
            if neuron.is_real_neuron and neuron.lif_neuron:
                self.real_neuron_manager.stimulate_neuron(neuron.id, current)
                print(f"⚡ 刺激真实神经元 {neuron.name} (电流: {current}nA)")
    
    def send_input_to_neuron(self, content: str, neuron_id: str = None) -> bool:
        if neuron_id is None:
            neuron = random.choice(self.neurons)
            neuron_id = neuron.id
        else:
            neuron = next((n for n in self.neurons if n.id == neuron_id), None)
        
        if neuron:
            self.io_manager.send_input(content, neuron.name, self.current_step)
            if neuron.is_real_neuron and neuron.lif_neuron:
                stimulus = len(content) * 0.1
                neuron.lif_neuron.inject_current(stimulus)
                neuron.satiety = min(1.0, neuron.satiety + 0.02)
                print(f"📨 输入 \"{content}\" 已发送给 {neuron.name}，饱腹度+2%")
            return True
        return False
    
    def get_neuron_outputs(self, count: int = 20) -> List:
        return self.io_manager.get_recent_outputs(count)
    
    def get_neuron_inputs(self, count: int = 20) -> List:
        return self.io_manager.get_recent_inputs(count)
    
    def step(self):
        if self.paused:
            return
        
        self.current_step += 1
        
        self.real_neuron_manager.step()
        
        for neuron in self.neurons:
            self._update_neuron(neuron)
        
        self._check_connections()
        
        self._check_divisions()
        
        self._check_deaths()
        
        self._generate_signals()
        
        self._collect_outputs()
        
        self.cluster_manager.update(self.neurons)
    
    def _update_neuron(self, neuron: Neuron):
        distance = neuron.position.magnitude()
        
        nutrition_rate = self._get_nutrition_rate(distance)
        neuron.satiety = min(1.0, neuron.satiety + nutrition_rate * 0.01)
        
        cluster_bonus = self.cluster_manager.get_cluster_bonus(neuron.id)
        self._move_neuron(neuron, distance, cluster_bonus)
        
        neuron.age += 1
        if len(neuron.synapses) == 0:
            neuron.steps_without_connection += 1
        
        if neuron.signal_cooldown > 0:
            neuron.signal_cooldown -= 1
    
    def _get_nutrition_rate(self, distance: float) -> float:
        R = self.R
        ratio = distance / R if R > 0 else 0
        
        if ratio <= 0.25:
            return 0.18
        elif ratio <= 0.5:
            return 0.10 + (0.5 - ratio) / 0.25 * 0.05
        elif ratio <= 1.0:
            return 0.03 + (1.0 - ratio) / 0.5 * 0.07
        return 0.03
    
    def _move_neuron(self, neuron: Neuron, distance: float, cluster_bonus: float):
        R = self.R
        
        direction = Vector3(
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ).normalized()
        
        if distance > R / 2:
            gravity_dir = -neuron.position.normalized()
            gravity_factor = 0.03 * (distance / R)
            direction = direction + gravity_dir * gravity_factor
        
        if neuron.satiety < 0.26:
            hunger_dir = -neuron.position.normalized()
            hunger_factor = (0.26 - neuron.satiety) * 2
            direction = direction + hunger_dir * hunger_factor
        
        speed = 1.0 * (1 + cluster_bonus)
        direction = direction.normalized()
        
        neuron.position = neuron.position + direction * speed
        
        new_distance = neuron.position.magnitude()
        if new_distance <= R / 2 and new_distance > 0:
            rotation_speed = 0.1 / new_distance
            x, z = neuron.position.x, neuron.position.z
            neuron.position.x = x * math.cos(rotation_speed) - z * math.sin(rotation_speed)
            neuron.position.z = x * math.sin(rotation_speed) + z * math.cos(rotation_speed)
        
        if neuron.position.magnitude() > R:
            neuron.position = neuron.position.normalized() * R * 0.99
    
    def _check_connections(self):
        for i, n1 in enumerate(self.neurons):
            for n2 in self.neurons[i+1:]:
                distance = (n1.position - n2.position).magnitude()
                
                if distance < self.CONNECTION_DISTANCE:
                    if n2.id not in n1.synapses:
                        if len(n1.synapses) < self.MAX_SYNAPSES and len(n2.synapses) < self.MAX_SYNAPSES:
                            n1.synapses.append(n2.id)
                            n2.synapses.append(n1.id)
                            n1.steps_without_connection = 0
                            n2.steps_without_connection = 0
                            print(f"[步骤{self.current_step}] 💫 {n1.name} 与 {n2.name} 连接!")
                            
                            if self.use_real_neurons and n1.is_real_neuron and n2.is_real_neuron:
                                weight = random.uniform(0.5, 1.5)
                                self.real_neuron_manager.connect_real_neurons(n1.id, n2.id, weight)
                                print(f"   → 真实突触连接建立 (权重: {weight:.2f})")
    
    def _check_divisions(self):
        new_neurons = []
        
        for neuron in self.neurons:
            if neuron.division_count >= self.MAX_DIVISIONS:
                continue
            
            should_divide = False
            if neuron.satiety >= 1.0 and neuron.steps_without_connection >= 100:
                should_divide = random.random() < 0.3
            elif neuron.satiety >= 0.5 and neuron.steps_without_connection >= 200:
                should_divide = random.random() < 0.2
            
            if should_divide:
                child = Neuron(
                    id=f"{neuron.id}_d{neuron.division_count}",
                    name=f"{neuron.name}子",
                    position=neuron.position + Vector3(
                        random.uniform(-20, 20),
                        random.uniform(-20, 20),
                        random.uniform(-20, 20)
                    ),
                    satiety=0.5,
                    is_division_child=True,
                    parent_id=neuron.id
                )
                new_neurons.append(child)
                neuron.division_count += 1
                neuron.satiety = 0.5
                print(f"[步骤{self.current_step}] 🔄 {neuron.name} 分裂产生 {child.name}!")
        
        self.neurons.extend(new_neurons)
    
    def _check_deaths(self):
        if self.current_step < self.DIVISION_MIN_STEP:
            to_remove = []
            for neuron in self.neurons:
                if neuron.is_division_child and neuron.parent_id:
                    parent_exists = any(n.id == neuron.parent_id for n in self.neurons)
                    if not parent_exists:
                        to_remove.append(neuron)
            
            for neuron in to_remove:
                self.neurons.remove(neuron)
                for n in self.neurons:
                    if neuron.id in n.synapses:
                        n.synapses.remove(neuron.id)
                print(f"[步骤{self.current_step}] ⚰️ {neuron.name} 消亡 (分裂子体，父体已不存在)")
    
    def _generate_signals(self):
        for neuron in self.neurons:
            if neuron.is_real_neuron and neuron.lif_neuron:
                if neuron.lif_neuron.last_spike_time == neuron.lif_neuron.age - 1:
                    if len(neuron.synapses) > 0:
                        target_id = random.choice(neuron.synapses)
                        target_neuron = next((n for n in self.neurons if n.id == target_id), None)
                        
                        self.signal_manager.generate_signal(
                            source_id=neuron.name,
                            target_id=target_neuron.name if target_neuron else target_id,
                            step=self.current_step,
                            is_cluster=neuron.id in self.cluster_manager.neuron_to_cluster
                        )
                        
                        if target_neuron and target_neuron.lif_neuron:
                            target_neuron.lif_neuron.receive_synaptic_input(
                                neuron.lif_neuron.spike_amplitude * 0.5,
                                is_excitatory=True
                            )
                            neuron.signal_cooldown = self.SIGNAL_INTERVAL
            else:
                if neuron.signal_cooldown > 0:
                    continue
                
                if len(neuron.synapses) > 0 and random.random() < 0.1:
                    target_id = random.choice(neuron.synapses)
                    is_cluster = neuron.id in self.cluster_manager.neuron_to_cluster
                    self.signal_manager.generate_signal(
                        source_id=neuron.name,
                        target_id=target_id,
                        step=self.current_step,
                        is_cluster=is_cluster
                    )
                    neuron.signal_cooldown = self.SIGNAL_INTERVAL
        
        for cluster_id, members in self.cluster_manager.clusters.items():
            if len(members) >= 3 and random.random() < 0.05:
                self.signal_manager.generate_signal(
                    source_id=cluster_id,
                    target_id=None,
                    step=self.current_step,
                    is_cluster=True
                )
    
    def _collect_outputs(self):
        for neuron in self.neurons:
            if neuron.is_real_neuron and neuron.lif_neuron:
                if neuron.lif_neuron.last_spike_time == neuron.lif_neuron.age - 1:
                    outputs = [
                        "脉冲信号",
                        f"膜电位{neuron.lif_neuron.membrane_potential:.1f}mV",
                        f"发放频率{neuron.lif_neuron.get_firing_rate():.1f}Hz",
                        "状态更新",
                        f"饱腹度{neuron.satiety:.0%}"
                    ]
                    output_content = random.choice(outputs)
                    self.io_manager.record_output(
                        neuron.name, 
                        output_content, 
                        self.current_step,
                        "发放触发"
                    )
                    neuron.satiety = min(1.0, neuron.satiety + 0.02)
    
    def get_stats(self) -> dict:
        connected = sum(1 for n in self.neurons if len(n.synapses) > 0)
        avg_satiety = sum(n.satiety for n in self.neurons) / len(self.neurons) if self.neurons else 0
        total_connections = sum(len(n.synapses) for n in self.neurons) // 2
        
        cluster_stats = self.cluster_manager.get_stats(len(self.neurons))
        signal_stats = self.signal_manager.get_stats()
        
        real_neuron_count = sum(1 for n in self.neurons if n.is_real_neuron)
        avg_membrane = 0.0
        total_spikes = 0
        
        if self.use_real_neurons:
            network_stats = self.real_neuron_manager.get_network_stats()
            total_spikes = network_stats.get('total_spikes', 0)
            avg_membrane = sum(n.get_membrane_potential() for n in self.neurons) / len(self.neurons) if self.neurons else 0
        
        return {
            'step': self.current_step,
            'neurons': len(self.neurons),
            'connected': connected,
            'connections': total_connections,
            'clusters': cluster_stats['cluster_count'],
            'single_neurons': cluster_stats['single_neurons'],
            'avg_satiety': round(avg_satiety, 2),
            'radius': self.R,
            'signals': signal_stats['total_signals'],
            'real_neurons': real_neuron_count,
            'total_spikes': total_spikes,
            'avg_membrane': round(avg_membrane, 2)
        }
    
    def get_real_signals(self, count: int = 50) -> List:
        if self.use_real_neurons:
            return self.real_neuron_manager.get_real_signals(count)
        return []
    
    def save(self) -> str:
        filename = os.path.join(self.data_dir, f"snapshot_{self.current_step:06d}.json")
        
        data = {
            'version': '0.0.1',
            'timestamp': datetime.now().isoformat(),
            'step': self.current_step,
            'radius': self.R,
            'use_real_neurons': self.use_real_neurons,
            'neurons': [n.to_dict() for n in self.neurons],
            'clusters': {cid: list(members) for cid, members in self.cluster_manager.clusters.items()},
            'recent_signals': [s.to_dict() for s in self.signal_manager.get_recent_signals(100)],
            'input_records': [r.to_dict() for r in self.io_manager.get_recent_inputs(100)],
            'output_records': [r.to_dict() for r in self.io_manager.get_recent_outputs(100)]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        input_log = os.path.join(self.data_dir, "input_history.json")
        with open(input_log, 'a', encoding='utf-8') as f:
            for r in self.io_manager.get_recent_inputs(10):
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        
        output_log = os.path.join(self.data_dir, "output_history.json")
        with open(output_log, 'a', encoding='utf-8') as f:
            for r in self.io_manager.get_recent_outputs(10):
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        
        print(f"💾 已保存: {filename}")
        print(f"   输入记录: {input_log}")
        print(f"   输出记录: {output_log}")
        
        cluster_count = self.cluster_manager.save_all_clusters(self.neurons, self.data_dir)
        if cluster_count > 0:
            print(f"   集群文件夹: {cluster_count}个")
        
        return filename
    
    def load_neuron_from_file(self, swc_path: str, alias: str = None) -> bool:
        if not os.path.exists(swc_path):
            print(f"❌ 文件不存在: {swc_path}")
            return False
        
        try:
            morph = SWCLoader.load_from_file(swc_path)
            neuron = self.real_neuron_manager.create_neuron_from_morphology(morph, self.R)
            
            if alias:
                neuron.name = alias
            
            self.neurons.append(neuron)
            
            target_dir = os.path.join(os.path.dirname(__file__), "neurons")
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, os.path.basename(swc_path))
            if swc_path != target_path:
                import shutil
                shutil.copy2(swc_path, target_path)
            
            print(f"✓ 投入神经元: {neuron.name} (ID: {neuron.id})")
            return True
        except Exception as e:
            print(f"❌ 加载神经元失败: {e}")
            return False
    
    def get_available_neurons(self, category: str = None) -> List[str]:
        base_path = "/Users/yan/projects/flywire--/神经元_已经下载分类"
        if not os.path.exists(base_path):
            return []
        
        swc_files = []
        if category:
            category_path = os.path.join(base_path, category)
            if os.path.exists(category_path):
                for f in os.listdir(category_path):
                    if f.endswith('.swc'):
                        swc_files.append(os.path.join(category_path, f))
        else:
            for root, dirs, files in os.walk(base_path):
                for f in files:
                    if f.endswith('.swc'):
                        swc_files.append(os.path.join(root, f))
        
        return swc_files[:100]


class Visualizer:
    def __init__(self, brain: BrainSpaceSimulator):
        self.brain = brain
        self.mode = '3d'
        self.update_interval = 30
        self.last_update = 0
        self.zoom_level = 1.0
        self.view_distance = 10.0
        self.signal_window_open = False
        self.signal_messages = []
        self.max_signal_messages = 50
        self.show_signal_panel = True
        self.input_buffer = ""
        self.selected_neuron_idx = 0
        
        if not HAS_MATPLOTLIB:
            raise RuntimeError("需要安装 matplotlib: pip install matplotlib numpy")
        
        self._setup_chinese_font()
        
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, wspace=0.15)
        
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.signal_ax = self.fig.add_subplot(122)
        self.signal_ax.axis('off')
        
        self._setup_buttons()
        
        self._set_window_normal()
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        self._print_help()
    
    def _setup_chinese_font(self):
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False
    
    def _setup_buttons(self):
        self.buttons = []
        
        btn_h = 0.035
        btn_y = 0.01
        btn_w = 0.045
        gap = 0.003
        group_gap = 0.015
        
        control_buttons = [
            ('▶⏸', ' ', '暂停/继续'),
            ('2D', '2', '切换2D'),
            ('3D', '3', '切换3D'),
        ]
        
        space_buttons = [
            ('⊖', '[', '缩小空间'),
            ('⊕', ']', '放大空间'),
            ('◀', '-', '减速'),
            ('▶', '=', '加速'),
        ]
        
        action_buttons = [
            ('投入', 'n', '投入神经元'),
            ('输入', 'e', '发送输入'),
            ('刺激', 't', '刺激神经元'),
            ('信号', 'i', '信号面板'),
        ]
        
        util_buttons = [
            ('保存', 's', '保存状态'),
            ('帮助', 'h', '帮助'),
            ('退出', 'q', '退出'),
        ]
        
        x = 0.02
        for label, key, tip in control_buttons:
            self._create_button(x, btn_y, btn_w, btn_h, label, key)
            x += btn_w + gap
        x += group_gap
        
        for label, key, tip in space_buttons:
            self._create_button(x, btn_y, btn_w, btn_h, label, key)
            x += btn_w + gap
        x += group_gap
        
        for label, key, tip in action_buttons:
            self._create_button(x, btn_y, btn_w, btn_h, label, key)
            x += btn_w + gap
        x += group_gap
        
        for label, key, tip in util_buttons:
            self._create_button(x, btn_y, btn_w, btn_h, label, key)
            x += btn_w + gap
    
    def _create_button(self, x, y, w, h, label, key):
        ax_btn = self.fig.add_axes([x, y, w, h])
        btn = widgets.Button(ax_btn, label)
        btn.on_clicked(lambda event, k=key: self._button_click(k))
        self.buttons.append(btn)
    
    def _button_click(self, key: str):
        class Event:
            def __init__(self, k):
                self.key = k
        self._on_key(Event(key))
    
    def _set_window_normal(self):
        try:
            manager = plt.get_current_fig_manager()
            
            if hasattr(manager, 'window'):
                window = manager.window
                
                if hasattr(window, 'attributes'):
                    window.attributes('-topmost', False)
                
                if hasattr(window, 'setWindowFlags'):
                    from PyQt5.QtCore import Qt
                    window.setWindowFlags(
                        window.windowFlags() & ~Qt.WindowStaysOnTopHint
                    )
                    window.show()
                
                if hasattr(window, '_qt'):
                    pass
                
                import sys
                if sys.platform == 'darwin':
                    try:
                        if hasattr(window, 'winId'):
                            view = window.contentView()
                            if view:
                                ns_win = view.window()
                                ns_win.setLevel_(0)
                    except Exception:
                        pass
                        
        except Exception:
            pass
    
    def _on_scroll(self, event):
        if event.button == 'up':
            self.view_distance = max(2.0, self.view_distance * 0.9)
        elif event.button == 'down':
            self.view_distance = min(50.0, self.view_distance * 1.1)
    
    def _print_help(self):
        print("\n" + "="*60)
        print("  球形脑空间模拟器 - 真实神经元版 v0.0.1")
        print("="*60)
        print("\n快捷键:")
        print("  空格: 暂停/继续")
        print("  2/3: 切换2D/3D模式")
        print("  [ ]: 缩小/放大空间")
        print("  - =: 减慢/加快速度")
        print("  H:   显示帮助窗口")
        print("  I:   显示/隐藏信号面板")
        print("  N:   投入新神经元 (从SWC文件)")
        print("  E:   向神经元发送输入 (汉字/文本)")
        print("  S:   保存状态")
        print("  T:   刺激随机神经元")
        print("  Q:   退出")
        print("\n信号面板显示:")
        print("  - 用户输入记录")
        print("  - 神经元输出记录")
        print("  - 神经元状态")
        print("  - 信号传输记录")
        print("\n鼠标滚轮: 缩放视图")
        print("="*60 + "\n")
    
    def show_help_window(self):
        help_text = """
┌─────────────────────────────────────────────────────────────┐
│                    球形脑空间模拟器 v0.0.1                     │
│                      快捷键帮助                              │
├─────────────────────────────────────────────────────────────┤
│  按键    │  功能              │  说明                       │
├─────────────────────────────────────────────────────────────┤
│  空格    │  暂停/继续         │  控制模拟运行               │
│  2/3     │  切换2D/3D模式     │  俯视图/点云显示            │
│  [ ]     │  缩小/放大空间     │  半径±10%                   │
│  - =     │  减慢/加快速度     │  时间比例调整               │
│  H       │  显示帮助          │  本窗口                     │
│  I       │  显示/隐藏信号    │  右侧信号面板               │
│  N       │  投入新神经元      │  从SWC文件加载              │
│  E       │  发送输入          │  向神经元发送汉字/文本      │
│  S       │  保存状态          │  保存到JSON文件             │
│  T       │  刺激神经元        │  刺激随机神经元             │
│  Q       │  退出              │  关闭程序                   │
├─────────────────────────────────────────────────────────────┤
│  信号面板显示：                                              │
│  - 用户输入记录: 汉字、词语、文本输入                        │
│  - 神经元输出记录: 神经元主动输出信息                        │
│  - 神经元状态: 膜电位、发放频率、饱腹度                      │
│  - 信号传输: 神经元间信号传递记录                            │
├─────────────────────────────────────────────────────────────┤
│  用户交互：                                                  │
│  N键 → 选择SWC文件 → 输入别名 → 投入脑空间                  │
│  E键 → 选择神经元 → 输入内容 → 发送给神经元                 │
│  发送输入后神经元饱腹度+2%                                   │
└─────────────────────────────────────────────────────────────┘
"""
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, help_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='center', 
                horizontalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        fig.canvas.mpl_connect('key_press_event', lambda e: plt.close(fig))
        fig.canvas.mpl_connect('button_press_event', lambda e: plt.close(fig))
        
        plt.show()
    
    def _switch_mode(self, new_mode: str):
        self.mode = new_mode
        self.ax.remove()
        
        for btn in self.buttons:
            btn.ax.remove()
        self.buttons = []
        
        if new_mode == '2d':
            self.ax = self.fig.add_subplot(121)
            print("切换到2D模式")
        else:
            self.ax = self.fig.add_subplot(121, projection='3d')
            print("切换到3D模式")
        
        self._setup_buttons()
    
    def _on_key(self, event):
        if event.key == ' ':
            self.brain.paused = not self.brain.paused
            status = "⏸️ 暂停" if self.brain.paused else "▶️ 继续"
            print(f"{status} 模拟")
        
        elif event.key == '2':
            if self.mode != '2d':
                self._switch_mode('2d')
        
        elif event.key == '3':
            if self.mode != '3d':
                self._switch_mode('3d')
        
        elif event.key == '[':
            old_R = self.brain.R
            self.brain.space_adjuster.shrink(0.9)
            self.brain.space_adjuster.adjust_positions(self.brain.neurons, old_R)
            print(f"[空间调整] 缩小: {old_R:.0f} → {self.brain.R:.0f}")
        
        elif event.key == ']':
            old_R = self.brain.R
            self.brain.space_adjuster.expand(1.1)
            self.brain.space_adjuster.adjust_positions(self.brain.neurons, old_R)
            print(f"[空间调整] 放大: {old_R:.0f} → {self.brain.R:.0f}")
        
        elif event.key == '-':
            self.brain.time_controller.slower()
            print(f"[速度调整] {self.brain.time_controller.get_info()}")
        
        elif event.key == '=':
            self.brain.time_controller.faster()
            print(f"[速度调整] {self.brain.time_controller.get_info()}")
        
        elif event.key == 'h':
            self.show_help_window()
        
        elif event.key == 'i':
            self.show_signal_panel = not self.show_signal_panel
            print(f"信号面板: {'显示' if self.show_signal_panel else '隐藏'}")
        
        elif event.key == 't':
            self.brain.stimulate_neuron()
        
        elif event.key == 's':
            self.brain.save()
        
        elif event.key == 'n':
            self._show_neuron_input_dialog()
        
        elif event.key == 'e':
            self._show_text_input_dialog()
        
        elif event.key == 'q':
            plt.close('all')
    
    def _show_neuron_input_dialog(self):
        print("\n" + "="*50)
        print("  投入新神经元")
        print("="*50)
        
        available = self.brain.get_available_neurons()
        if not available:
            print("❌ 没有找到可用的SWC文件")
            print("请确保目录存在: /Users/yan/projects/flywire--/神经元_已经下载分类")
            return
        
        print(f"\n找到 {len(available)} 个可用神经元文件")
        print("前10个文件:")
        for i, path in enumerate(available[:10]):
            print(f"  {i+1}. {os.path.basename(path)}")
        
        try:
            choice = input("\n输入序号(1-10)或回车随机选择: ").strip()
            if choice == "":
                selected = random.choice(available[:10])
            else:
                idx = int(choice) - 1
                if 0 <= idx < min(10, len(available)):
                    selected = available[idx]
                else:
                    print("❌ 无效选择")
                    return
            
            alias = input("输入神经元别名(回车跳过): ").strip()
            self.brain.load_neuron_from_file(selected, alias if alias else None)
        except (ValueError, KeyboardInterrupt):
            print("\n取消投入")
    
    def _show_text_input_dialog(self):
        print("\n" + "="*50)
        print("  向神经元发送输入")
        print("="*50)
        
        neurons = self.brain.neurons[:10]
        print("\n当前神经元:")
        for i, n in enumerate(neurons):
            print(f"  {i+1}. {n.name} (饱腹度: {n.satiety:.0%})")
        
        try:
            choice = input("\n输入序号(1-10)或回车随机选择: ").strip()
            if choice == "":
                neuron = random.choice(neurons)
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(neurons):
                    neuron = neurons[idx]
                else:
                    print("❌ 无效选择")
                    return
            
            content = input(f"输入要发送给 {neuron.name} 的内容: ").strip()
            if content:
                self.brain.send_input_to_neuron(content, neuron.id)
            else:
                print("❌ 内容为空")
        except (ValueError, KeyboardInterrupt):
            print("\n取消输入")
    
    def should_update(self) -> bool:
        if self.brain.current_step - self.last_update >= self.update_interval:
            self.last_update = self.brain.current_step
            return True
        return False
    
    def draw(self):
        if not self.should_update():
            return
        
        self.ax.clear()
        R = self.brain.R * self.zoom_level
        
        if self.mode == '2d':
            self._draw_2d(R)
        else:
            self._draw_3d(R)
        
        if self.show_signal_panel:
            self._draw_signal_panel()
        
        plt.pause(0.001)
    
    def _draw_signal_panel(self):
        self.signal_ax.clear()
        self.signal_ax.axis('off')
        
        self.signal_ax.set_facecolor('#f8f8f8')
        
        stats = self.brain.get_stats()
        io_stats = self.brain.io_manager.get_stats()
        speed_info = self.brain.time_controller.speed_name
        status = "⏸ 暂停" if self.brain.paused else "▶ 运行"
        
        header = f"═══ 神经元信号监视器 ═══\n"
        header += f"步骤: {stats['step']:>6} | {status} | {speed_info}\n"
        header += f"神经元: {stats['neurons']:>3} | 连接: {stats['connections']:>3} | 集群: {stats['clusters']:>2}\n"
        header += "═" * 32
        
        self.signal_ax.text(0.02, 0.98, header, transform=self.signal_ax.transAxes,
                           fontsize=8, family='monospace', fontweight='bold',
                           verticalalignment='top', color='#333')
        
        y_pos = 0.78
        line_height = 0.028
        
        self.signal_ax.text(0.02, y_pos, '┌─ 用户输入记录 ─────────┐', 
                           transform=self.signal_ax.transAxes,
                           fontsize=7, family='monospace', color='#8B0000')
        y_pos -= line_height
        
        inputs = self.brain.get_neuron_inputs(4)
        if inputs:
            for inp in inputs[-4:]:
                text = f"│[{inp.step:5d}] \"{inp.content[:6]}\"→{inp.target_neuron[:6]}"
                self.signal_ax.text(0.02, y_pos, text, transform=self.signal_ax.transAxes,
                                   fontsize=6, family='monospace', color='#A0522D')
                y_pos -= line_height
        else:
            self.signal_ax.text(0.02, y_pos, '│ 暂无输入 (按E发送)', 
                               transform=self.signal_ax.transAxes,
                               fontsize=6, family='monospace', color='gray')
            y_pos -= line_height
        self.signal_ax.text(0.02, y_pos, '└' + '─' * 24 + '┘', 
                           transform=self.signal_ax.transAxes,
                           fontsize=7, family='monospace', color='#8B0000')
        y_pos -= line_height * 1.2
        
        self.signal_ax.text(0.02, y_pos, '┌─ 神经元输出记录 ───────┐', 
                           transform=self.signal_ax.transAxes,
                           fontsize=7, family='monospace', color='#006400')
        y_pos -= line_height
        
        outputs = self.brain.get_neuron_outputs(4)
        if outputs:
            for out in outputs[-4:]:
                text = f"│[{out.step:5d}] {out.source_neuron[:6]}:{out.content[:8]}"
                self.signal_ax.text(0.02, y_pos, text, transform=self.signal_ax.transAxes,
                                   fontsize=6, family='monospace', color='#228B22')
                y_pos -= line_height
        else:
            self.signal_ax.text(0.02, y_pos, '│ 暂无输出', 
                               transform=self.signal_ax.transAxes,
                               fontsize=6, family='monospace', color='gray')
            y_pos -= line_height
        self.signal_ax.text(0.02, y_pos, '└' + '─' * 24 + '┘', 
                           transform=self.signal_ax.transAxes,
                           fontsize=7, family='monospace', color='#006400')
        y_pos -= line_height * 1.2
        
        self.signal_ax.text(0.02, y_pos, '┌─ 神经元状态 ───────────┐', 
                           transform=self.signal_ax.transAxes,
                           fontsize=7, family='monospace', color='#00008B')
        y_pos -= line_height
        
        active_neurons = [n for n in self.brain.neurons if n.lif_neuron][:4]
        if active_neurons:
            for neuron in active_neurons:
                lif = neuron.lif_neuron
                pot = lif.membrane_potential
                rate = lif.get_firing_rate()
                status_icon = "🔥" if lif.refractory_timer > 0 else ("⚡" if pot > -55 else "💤")
                text = f"│{neuron.name[:6]:6s}|{pot:6.1f}mV|{rate:4.1f}Hz|{status_icon}"
                self.signal_ax.text(0.02, y_pos, text, transform=self.signal_ax.transAxes,
                                   fontsize=6, family='monospace')
                y_pos -= line_height
        else:
            self.signal_ax.text(0.02, y_pos, '│ 无活跃神经元', 
                               transform=self.signal_ax.transAxes,
                               fontsize=6, family='monospace', color='gray')
            y_pos -= line_height
        self.signal_ax.text(0.02, y_pos, '└' + '─' * 24 + '┘', 
                           transform=self.signal_ax.transAxes,
                           fontsize=7, family='monospace', color='#00008B')
        y_pos -= line_height * 1.2
        
        self.signal_ax.text(0.02, y_pos, '┌─ 信号传输 ─────────────┐', 
                           transform=self.signal_ax.transAxes,
                           fontsize=7, family='monospace', color='#4B0082')
        y_pos -= line_height
        
        signals = self.brain.signal_manager.get_recent_signals(4)
        if signals:
            for s in signals[-4:]:
                arrow = '→' if s.target_id else '⇒'
                target = str(s.target_id)[:5] if s.target_id else '广播'
                text = f"│[{s.step:5d}] {str(s.source_id)[:5]:5s}{arrow}{target}"
                self.signal_ax.text(0.02, y_pos, text, transform=self.signal_ax.transAxes,
                                   fontsize=6, family='monospace', color='#8A2BE2')
                y_pos -= line_height
        else:
            self.signal_ax.text(0.02, y_pos, '│ 暂无信号', 
                               transform=self.signal_ax.transAxes,
                               fontsize=6, family='monospace', color='gray')
            y_pos -= line_height
        self.signal_ax.text(0.02, y_pos, '└' + '─' * 24 + '┘', 
                           transform=self.signal_ax.transAxes,
                           fontsize=7, family='monospace', color='#4B0082')
        y_pos -= line_height * 1.5
        
        self.signal_ax.text(0.02, y_pos, '─────────────────────────', 
                           transform=self.signal_ax.transAxes,
                           fontsize=6, family='monospace', color='gray')
        y_pos -= line_height
        self.signal_ax.text(0.02, y_pos, '输入:按E | 投入:按N | 保存:按S', 
                           transform=self.signal_ax.transAxes,
                           fontsize=6, family='monospace', color='#666')
    
    def _draw_2d(self, R: float):
        actual_R = self.brain.R
        for r, color in [(actual_R, 'blue'), (actual_R/2, 'green'), (actual_R/4, 'red')]:
            circle = plt.Circle((0, 0), r, fill=False, color=color, linewidth=1.5)
            self.ax.add_patch(circle)
        
        for neuron in self.brain.neurons:
            x, y = neuron.position.x, neuron.position.y
            
            if neuron.id in self.brain.cluster_manager.neuron_to_cluster:
                color = 'orange'
                size = 50
                marker = 's'
            else:
                color = plt.cm.RdYlGn(neuron.satiety)
                size = 30
                marker = 'o'
            
            self.ax.scatter(x, y, c=[color], s=size, marker=marker, alpha=0.8)
        
        for neuron in self.brain.neurons:
            for synapse_id in neuron.synapses:
                other = next((n for n in self.brain.neurons if n.id == synapse_id), None)
                if other and neuron.id < synapse_id:
                    self.ax.plot([neuron.position.x, other.position.x],
                                [neuron.position.y, other.position.y],
                                'gray', alpha=0.3, linewidth=0.5)
        
        stats = self.brain.get_stats()
        speed_info = self.brain.time_controller.speed_name
        self.ax.set_title(
            f"球形脑空间 (2D) | 步骤:{stats['step']} | "
            f"单:{stats['single_neurons']} | 集群:{stats['clusters']} | "
            f"信号:{stats['signals']} | 速度:{speed_info}",
            fontsize=10
        )
        
        self.ax.set_xlim(-R * 1.1, R * 1.1)
        self.ax.set_ylim(-R * 1.1, R * 1.1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
    
    def _draw_3d(self, R: float):
        actual_R = self.brain.R
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 8)
        
        for r, color, alpha in [(actual_R, 'blue', 0.1), (actual_R/2, 'green', 0.15)]:
            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)
        
        xs, ys, zs, colors, sizes = [], [], [], [], []
        for neuron in self.brain.neurons:
            xs.append(neuron.position.x)
            ys.append(neuron.position.y)
            zs.append(neuron.position.z)
            
            if neuron.id in self.brain.cluster_manager.neuron_to_cluster:
                colors.append(np.array([1.0, 0.6, 0.0, 0.7]))
                sizes.append(40)
            else:
                colors.append(np.array(plt.cm.RdYlGn(neuron.satiety)))
                sizes.append(20)
        
        self.ax.scatter(xs, ys, zs, c=np.array(colors), s=sizes, alpha=0.7)
        
        for neuron in self.brain.neurons:
            for synapse_id in neuron.synapses:
                other = next((n for n in self.brain.neurons if n.id == synapse_id), None)
                if other and neuron.id < synapse_id:
                    self.ax.plot([neuron.position.x, other.position.x],
                                [neuron.position.y, other.position.y],
                                [neuron.position.z, other.position.z],
                                'gray', alpha=0.3, linewidth=0.5)
        
        stats = self.brain.get_stats()
        speed_info = self.brain.time_controller.speed_name
        self.ax.set_title(
            f"球形脑空间 (3D) | 步骤:{stats['step']} | "
            f"单:{stats['single_neurons']} | 集群:{stats['clusters']} | "
            f"信号:{stats['signals']} | 速度:{speed_info}",
            fontsize=10
        )
        
        self.ax.set_xlim(-R * 1.1, R * 1.1)
        self.ax.set_ylim(-R * 1.1, R * 1.1)
        self.ax.set_zlim(-R * 1.1, R * 1.1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        self.ax.dist = self.view_distance


def main():
    parser = argparse.ArgumentParser(description='球形脑空间模拟器 - 真实神经元版')
    parser.add_argument('--category', type=str, default=None,
                        help='指定神经元分类目录')
    parser.add_argument('--count', type=int, default=10,
                        help='神经元数量（默认10个）')
    parser.add_argument('--radius', type=int, default=1000,
                        help='球形空间半径（默认1000）')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  球形脑空间模拟器 - 真实神经元版本 v0.0.1")
    print("="*60)
    
    print("\n模式: 真实SWC神经元")
    print(f"神经元数量: {args.count}")
    if args.category:
        print(f"神经元分类: {args.category}")
    
    print("\n初始化...")
    
    brain = BrainSpaceSimulator(
        R=args.radius, 
        initial_count=args.count,
        category=args.category
    )
    
    if HAS_MATPLOTLIB:
        viz = Visualizer(brain)
        
        print(f"\n开始模拟... (当前速度: {brain.time_controller.get_info()})")
        print(f"分裂规则: {brain.DIVISION_MIN_STEP}步前分裂的子体会消亡")
        print(f"按 I 键显示/隐藏信号面板")
        print(f"按 T 键刺激随机神经元")
        print(f"鼠标滚轮: 缩放视图\n")
        
        try:
            window_fix_counter = 0
            while plt.fignum_exists(viz.fig.number):
                if not brain.paused:
                    steps = brain.time_controller.steps_per_frame
                    for _ in range(steps):
                        brain.step()
                viz.draw()
                
                window_fix_counter += 1
                if window_fix_counter >= 50:
                    viz._set_window_normal()
                    window_fix_counter = 0
        except KeyboardInterrupt:
            print("\n\n模拟已停止")
    else:
        print("\n纯文本模式运行（每100步输出一次状态）...\n")
        
        try:
            while True:
                brain.step()
                
                if brain.current_step % 100 == 0:
                    stats = brain.get_stats()
                    print(f"[步骤{stats['step']}] "
                          f"神经元:{stats['neurons']} | "
                          f"连接:{stats['connections']} | "
                          f"集群:{stats['clusters']} | "
                          f"信号:{stats['signals']} | "
                          f"平均饱腹:{stats['avg_satiety']}")
        except KeyboardInterrupt:
            print("\n\n模拟已停止")
            stats = brain.get_stats()
            print(f"\n最终状态: {stats}")


if __name__ == "__main__":
    main()
