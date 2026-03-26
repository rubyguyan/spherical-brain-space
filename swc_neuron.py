#!/usr/bin/env python3
"""
SWC神经元模块 - 真实神经元支持
================================
功能：
├── SWC文件解析（标准神经元形态格式）
├── LIF神经元模型（Leaky Integrate-and-Fire）
├── 真实神经元信号发送/接收
├── 突触连接管理
└── 神经元文件自动复制

SWC格式说明：
- 每行: 点编号 标签 X Y Z 半径 父节点
- 标签: 0=未定义, 1=胞体(soma), 5=分叉点, 6=端点
"""

import os
import shutil
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque


@dataclass
class SWCPoint:
    index: int
    label: int
    x: float
    y: float
    z: float
    radius: float
    parent: int
    
    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    @property
    def is_soma(self) -> bool:
        return self.label == 1
    
    @property
    def is_fork(self) -> bool:
        return self.label == 5
    
    @property
    def is_endpoint(self) -> bool:
        return self.label == 6


@dataclass
class NeuronMorphology:
    neuron_id: str
    name: str
    points: List[SWCPoint] = field(default_factory=list)
    soma_position: Optional[Tuple[float, float, float]] = None
    total_length: float = 0.0
    branch_count: int = 0
    source_file: str = ""
    
    def get_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if not self.points:
            return ((0, 0, 0), (0, 0, 0))
        
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        zs = [p.z for p in self.points]
        
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    
    def get_center(self) -> Tuple[float, float, float]:
        if not self.points:
            return (0, 0, 0)
        
        bounds = self.get_bounds()
        return (
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
            (bounds[0][2] + bounds[1][2]) / 2
        )
    
    def get_size(self) -> float:
        bounds = self.get_bounds()
        return max(
            bounds[1][0] - bounds[0][0],
            bounds[1][1] - bounds[0][1],
            bounds[1][2] - bounds[0][2]
        )


class SWCLoader:
    SWC_DIR = "/Users/yan/projects/flywire--/神经元_已经下载分类"
    TARGET_DIR = "/Users/yan/projects/flywire--/球形脑空间/neurons"
    
    FAFB_X_MIN = 350000.0
    FAFB_X_MAX = 550000.0
    FAFB_Y_MIN = 80000.0
    FAFB_Y_MAX = 220000.0
    FAFB_Z_MIN = 20000.0
    FAFB_Z_MAX = 240000.0
    
    @staticmethod
    def transform_fafb_to_sphere(x: float, y: float, z: float, 
                                  sphere_radius: float = 1000.0) -> Tuple[float, float, float]:
        norm_x = max(0, min(1, (x - SWCLoader.FAFB_X_MIN) / (SWCLoader.FAFB_X_MAX - SWCLoader.FAFB_X_MIN)))
        norm_y = max(0, min(1, (y - SWCLoader.FAFB_Y_MIN) / (SWCLoader.FAFB_Y_MAX - SWCLoader.FAFB_Y_MIN)))
        norm_z = max(0, min(1, (z - SWCLoader.FAFB_Z_MIN) / (SWCLoader.FAFB_Z_MAX - SWCLoader.FAFB_Z_MIN)))
        
        sphere_x = (norm_x - 0.5) * 2 * sphere_radius * 0.6
        sphere_y = (norm_y - 0.5) * 2 * sphere_radius * 0.6
        sphere_z = (norm_z - 0.5) * 2 * sphere_radius * 0.6
        
        return (sphere_x, sphere_y, sphere_z)
    
    @staticmethod
    def parse_swc_file(filepath: str) -> NeuronMorphology:
        points = []
        neuron_id = ""
        name = os.path.basename(filepath).replace('.swc', '')
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('# Meta:'):
                    try:
                        import json
                        meta_start = line.find('{')
                        if meta_start != -1:
                            meta = json.loads(line[meta_start:])
                            neuron_id = meta.get('id', '')
                            name = meta.get('name', name)
                    except:
                        pass
                    continue
                
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        point = SWCPoint(
                            index=int(parts[0]),
                            label=int(parts[1]),
                            x=float(parts[2]),
                            y=float(parts[3]),
                            z=float(parts[4]),
                            radius=float(parts[5]),
                            parent=int(parts[6])
                        )
                        points.append(point)
                    except ValueError:
                        continue
        
        soma_points = [p for p in points if p.is_soma]
        soma_pos = soma_points[0].position if soma_points else None
        
        morphology = NeuronMorphology(
            neuron_id=neuron_id,
            name=name,
            points=points,
            soma_position=soma_pos,
            source_file=filepath
        )
        
        morphology.total_length = SWCLoader._calculate_total_length(points)
        morphology.branch_count = sum(1 for p in points if p.is_fork)
        
        return morphology
    
    @staticmethod
    def _calculate_total_length(points: List[SWCPoint]) -> float:
        point_map = {p.index: p for p in points}
        total = 0.0
        
        for point in points:
            if point.parent > 0 and point.parent in point_map:
                parent = point_map[point.parent]
                dx = point.x - parent.x
                dy = point.y - parent.y
                dz = point.z - parent.z
                total += math.sqrt(dx*dx + dy*dy + dz*dz)
        
        return total
    
    @staticmethod
    def get_available_categories() -> List[str]:
        categories = []
        if os.path.exists(SWCLoader.SWC_DIR):
            for item in os.listdir(SWCLoader.SWC_DIR):
                item_path = os.path.join(SWCLoader.SWC_DIR, item)
                if os.path.isdir(item_path) and item.startswith('神经元_'):
                    categories.append(item)
        return categories
    
    @staticmethod
    def get_swc_files_in_category(category: str) -> List[str]:
        category_path = os.path.join(SWCLoader.SWC_DIR, category)
        if not os.path.exists(category_path):
            return []
        
        swc_files = []
        for item in os.listdir(category_path):
            if item.endswith('.swc'):
                swc_files.append(os.path.join(category_path, item))
        return swc_files
    
    @staticmethod
    def copy_neuron_to_target(swc_file: str) -> str:
        os.makedirs(SWCLoader.TARGET_DIR, exist_ok=True)
        
        filename = os.path.basename(swc_file)
        target_path = os.path.join(SWCLoader.TARGET_DIR, filename)
        
        if not os.path.exists(target_path):
            shutil.copy2(swc_file, target_path)
            print(f"✓ 复制神经元文件: {filename}")
        
        return target_path
    
    @staticmethod
    def load_random_neurons(count: int, category: str = None) -> List[NeuronMorphology]:
        if category:
            categories = [category]
        else:
            categories = SWCLoader.get_available_categories()
        
        if not categories:
            print("警告: 未找到SWC神经元文件目录")
            return []
        
        all_files = []
        for cat in categories:
            all_files.extend(SWCLoader.get_swc_files_in_category(cat))
        
        if not all_files:
            print("警告: 未找到SWC文件")
            return []
        
        selected = random.sample(all_files, min(count, len(all_files)))
        
        morphologies = []
        for swc_file in selected:
            try:
                morph = SWCLoader.parse_swc_file(swc_file)
                SWCLoader.copy_neuron_to_target(swc_file)
                morphologies.append(morph)
            except Exception as e:
                print(f"警告: 加载 {swc_file} 失败: {e}")
        
        return morphologies


class LIFNeuron:
    """
    Leaky Integrate-and-Fire 神经元模型
    模拟真实神经元的电生理特性
    """
    
    RESTING_POTENTIAL = -70.0
    THRESHOLD_POTENTIAL = -55.0
    RESET_POTENTIAL = -75.0
    SPIKE_AMPLITUDE = 40.0
    
    def __init__(self, morphology: NeuronMorphology = None, neuron_id: str = None):
        self.morphology = morphology
        self.neuron_id = neuron_id or (morphology.neuron_id if morphology else "unknown")
        self.name = morphology.name if morphology else neuron_id
        
        self.membrane_potential = self.RESTING_POTENTIAL
        self.membrane_resistance = random.uniform(80, 150)
        self.membrane_capacitance = random.uniform(0.5, 1.5)
        self.tau_m = self.membrane_resistance * self.membrane_capacitance
        
        self.refractory_period = random.uniform(1.5, 3.0)
        self.refractory_timer = 0.0
        
        self.excitatory_synapses: List[str] = []
        self.inhibitory_synapses: List[str] = []
        
        self.input_current = 0.0
        self.spike_history: deque = deque(maxlen=100)
        self.last_spike_time = -1000.0
        
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        
        self.satiety = 0.5
        self.age = 0
        self.steps_without_connection = 0
        
        if morphology and morphology.soma_position:
            self.original_soma = morphology.soma_position
    
    def inject_current(self, current: float, duration: float = 1.0):
        self.input_current += current * duration
    
    def receive_synaptic_input(self, weight: float, is_excitatory: bool = True):
        if is_excitatory:
            self.input_current += weight * 0.5
        else:
            self.input_current -= weight * 0.3
    
    def update(self, dt: float = 1.0, external_input: float = 0.0) -> bool:
        spiked = False
        
        if self.refractory_timer > 0:
            self.refractory_timer -= dt
            self.membrane_potential = self.RESET_POTENTIAL
            return False
        
        total_input = self.input_current + external_input
        
        dV = (-(self.membrane_potential - self.RESTING_POTENTIAL) + 
              self.membrane_resistance * total_input) / self.tau_m
        
        self.membrane_potential += dV * dt
        
        self.input_current *= 0.9
        
        if self.membrane_potential >= self.THRESHOLD_POTENTIAL:
            spiked = True
            self.spike_history.append(self.age)
            self.last_spike_time = self.age
            self.membrane_potential = self.RESET_POTENTIAL
            self.refractory_timer = self.refractory_period
        
        return spiked
    
    def get_firing_rate(self, window: float = 100.0) -> float:
        if not self.spike_history:
            return 0.0
        
        recent_spikes = [t for t in self.spike_history if self.age - t <= window]
        return len(recent_spikes) / window if window > 0 else 0.0
    
    def get_state(self) -> dict:
        return {
            'neuron_id': self.neuron_id,
            'name': self.name,
            'membrane_potential': self.membrane_potential,
            'firing_rate': self.get_firing_rate(),
            'is_refractory': self.refractory_timer > 0,
            'satiety': self.satiety,
            'age': self.age,
            'synapse_count': len(self.excitatory_synapses) + len(self.inhibitory_synapses)
        }


class NeuronSignal:
    ACTION_POTENTIAL = "动作电位"
    GRADED_POTENTIAL = "等级电位"
    SYNAPTIC_TRANSMISSION = "突触传递"
    NEUROMODULATION = "神经调控"
    
    def __init__(self, source_id: str, signal_type: str, amplitude: float, 
                 target_id: str = None, timestamp: int = 0):
        self.source_id = source_id
        self.signal_type = signal_type
        self.amplitude = amplitude
        self.target_id = target_id
        self.timestamp = timestamp
        self.frequency = 0.0
        self.duration = 1.0
    
    def __str__(self) -> str:
        target = self.target_id or "广播"
        return f"[{self.timestamp}] {self.source_id} → {target}: {self.signal_type} ({self.amplitude:.2f}mV)"


class NeuronNetwork:
    def __init__(self):
        self.neurons: Dict[str, LIFNeuron] = {}
        self.connections: Dict[str, List[Tuple[str, float, bool]]] = {}
        self.signal_history: deque = deque(maxlen=500)
        self.current_step = 0
    
    def add_neuron(self, neuron: LIFNeuron):
        self.neurons[neuron.neuron_id] = neuron
        if neuron.neuron_id not in self.connections:
            self.connections[neuron.neuron_id] = []
    
    def connect_neurons(self, source_id: str, target_id: str, 
                        weight: float = 1.0, is_excitatory: bool = True):
        if source_id in self.neurons and target_id in self.neurons:
            self.connections[source_id].append((target_id, weight, is_excitatory))
            
            source = self.neurons[source_id]
            target = self.neurons[target_id]
            
            if is_excitatory:
                if target_id not in source.excitatory_synapses:
                    source.excitatory_synapses.append(target_id)
            else:
                if target_id not in source.inhibitory_synapses:
                    source.inhibitory_synapses.append(target_id)
            
            return True
        return False
    
    def disconnect_neurons(self, source_id: str, target_id: str):
        if source_id in self.connections:
            self.connections[source_id] = [
                (t, w, e) for t, w, e in self.connections[source_id] if t != target_id
            ]
        
        source = self.neurons.get(source_id)
        if source:
            if target_id in source.excitatory_synapses:
                source.excitatory_synapses.remove(target_id)
            if target_id in source.inhibitory_synapses:
                source.inhibitory_synapses.remove(target_id)
    
    def step(self, dt: float = 1.0):
        self.current_step += 1
        
        spikes = {}
        for neuron_id, neuron in self.neurons.items():
            neuron.age += 1
            external_input = self._calculate_external_input(neuron)
            spiked = neuron.update(dt, external_input)
            
            if spiked:
                spikes[neuron_id] = neuron
                signal = NeuronSignal(
                    source_id=neuron_id,
                    signal_type=NeuronSignal.ACTION_POTENTIAL,
                    amplitude=neuron.SPIKE_AMPLITUDE,
                    timestamp=self.current_step
                )
                self.signal_history.append(signal)
        
        for source_id, source_neuron in spikes.items():
            for target_id, weight, is_excitatory in self.connections.get(source_id, []):
                if target_id in self.neurons:
                    self.neurons[target_id].receive_synaptic_input(weight, is_excitatory)
                    
                    signal = NeuronSignal(
                        source_id=source_id,
                        signal_type=NeuronSignal.SYNAPTIC_TRANSMISSION,
                        amplitude=weight * (1 if is_excitatory else -1),
                        target_id=target_id,
                        timestamp=self.current_step
                    )
                    self.signal_history.append(signal)
    
    def _calculate_external_input(self, neuron: LIFNeuron) -> float:
        base_input = 0.0
        
        if neuron.satiety > 0.7:
            base_input += 0.5
        elif neuron.satiety < 0.3:
            base_input -= 0.3
        
        synapse_count = len(neuron.excitatory_synapses) + len(neuron.inhibitory_synapses)
        if synapse_count > 0:
            base_input += 0.2
        
        noise = random.gauss(0, 0.1)
        
        return base_input + noise
    
    def stimulate_neuron(self, neuron_id: str, current: float):
        if neuron_id in self.neurons:
            self.neurons[neuron_id].inject_current(current)
            
            signal = NeuronSignal(
                source_id="外部刺激",
                signal_type=NeuronSignal.NEUROMODULATION,
                amplitude=current,
                target_id=neuron_id,
                timestamp=self.current_step
            )
            self.signal_history.append(signal)
    
    def get_recent_signals(self, count: int = 50) -> List[NeuronSignal]:
        return list(self.signal_history)[-count:]
    
    def get_neuron_state(self, neuron_id: str) -> Optional[dict]:
        if neuron_id in self.neurons:
            return self.neurons[neuron_id].get_state()
        return None
    
    def get_network_stats(self) -> dict:
        total_spikes = sum(len(n.spike_history) for n in self.neurons.values())
        active_neurons = sum(1 for n in self.neurons.values() if len(n.spike_history) > 0)
        total_connections = sum(len(conns) for conns in self.connections.values())
        
        return {
            'total_neurons': len(self.neurons),
            'active_neurons': active_neurons,
            'total_connections': total_connections,
            'total_spikes': total_spikes,
            'current_step': self.current_step,
            'signals_in_history': len(self.signal_history)
        }


def create_real_neuron_network(neuron_count: int = 10, 
                                category: str = None) -> NeuronNetwork:
    morphologies = SWCLoader.load_random_neurons(neuron_count, category)
    
    network = NeuronNetwork()
    
    for i, morph in enumerate(morphologies):
        neuron = LIFNeuron(morphology=morph)
        
        center = morph.get_center()
        size = morph.get_size()
        scale = 1000.0 / max(size, 1.0)
        
        neuron.position = [
            center[0] * scale * 0.001,
            center[1] * scale * 0.001,
            center[2] * scale * 0.001
        ]
        
        network.add_neuron(neuron)
    
    print(f"✓ 创建真实神经元网络: {len(morphologies)} 个神经元")
    print(f"✓ 神经元文件已复制到: {SWCLoader.TARGET_DIR}")
    
    return network


if __name__ == "__main__":
    print("=" * 60)
    print("SWC神经元模块测试")
    print("=" * 60)
    
    categories = SWCLoader.get_available_categories()
    print(f"\n可用的神经元分类: {len(categories)}")
    for cat in categories:
        files = SWCLoader.get_swc_files_in_category(cat)
        print(f"  - {cat}: {len(files)} 个神经元")
    
    print("\n加载测试神经元...")
    network = create_real_neuron_network(5)
    
    print("\n网络统计:")
    stats = network.get_network_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n运行10步模拟...")
    for i in range(10):
        network.step()
        signals = network.get_recent_signals(5)
        if signals:
            for s in signals[-3:]:
                print(f"  {s}")
    
    print("\n✓ 测试完成")
