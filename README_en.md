# Spherical Brain Space Simulator

English Version | [中文版](README.md)

> **Original Declaration**: This concept was first proposed and publicly released by rubyguyan in March 2026.
> 
> For detailed concept description, see: [IDEA.md](IDEA.md)

A brain space simulator based on **FlyWire/FlyConnectome** real neuron data.

## Introduction

This project uses real neuron morphology data (SWC format) from the fruit fly brain to simulate neural networks in a spherical 3D space. Neurons automatically form synaptic connections through collisions, simulating real neural network activity.

## Core Innovations

- **Spherical Space Constraint**: Neurons are constrained within a spherical space, forming a natural spatial gradient
- **Real Neuron Data**: Uses real neuron morphology from the FlyWire project
- **Collision-Driven Connections**: Neurons automatically form synapses through physical collisions
- **Nutrient Supply System**: Nutrients are abundant at the center, scarce at the edges
- **Neuron Lifecycle**: Division, death, and dynamic balance

## Data Sources

- **FlyWire** - Fruit fly whole-brain connectome project
- **FAFB** (Full Adult Fly Brain) - Adult fruit fly whole-brain dataset
- Neuron IDs from the FlyWire database

## Features

- Real SWC neuron file loading
- LIF (Leaky Integrate-and-Fire) neuron model
- Neuron signal sending/receiving interface
- Automatic synaptic connection through collision
- Nutrient supply system (spatial gradient)
- 2D/3D visualization switching
- Dynamic space size adjustment
- Time scale speed control
- Automatic cluster identification
- Signal monitoring panel

## Installation

```bash
pip install matplotlib numpy
```

## Usage

```bash
python brain_space.py
```

### Command Line Options

```bash
python brain_space.py --count 10        # Number of neurons
python brain_space.py --category name   # Specify neuron category
python brain_space.py --radius 1000     # Spherical space radius
```

## Keyboard Shortcuts

| Key | Function |
|-----|----------|
| Space | Pause/Resume |
| 2/3 | Switch 2D/3D mode |
| [ ] | Shrink/Expand space |
| - = | Slow down/Speed up |
| H | Show help window |
| I | Show/Hide signal panel |
| N | Add new neuron |
| E | Send input to neuron |
| S | Save state |
| T | Stimulate random neuron |
| Q | Quit |

## File Structure

```
spherical-brain-space/
├── brain_space.py      # Main program
├── swc_neuron.py       # SWC neuron module
├── IDEA.md             # Concept description
├── README.md           # Chinese documentation
├── README_en.md        # English documentation
├── neurons/            # Neuron SWC files
│   └── *.swc          # Real neuron morphology data
├── docs/               # Documentation
└── brain_space_data/   # Runtime data
```

## SWC File Format

Standard SWC neuron morphology format:
- Each line: Point ID, Label, X, Y, Z, Radius, Parent
- Labels: 0=undefined, 1=soma, 5=fork point, 6=end point

## Technical Details

### Coordinate Transformation

FAFB coordinates (nanometers) to spherical space coordinates:
- X: 350000-550000 nm
- Y: 80000-220000 nm  
- Z: 20000-240000 nm

### LIF Neuron Model Parameters

- Resting potential: -70 mV
- Threshold potential: -55 mV
- Reset potential: -75 mV
- Refractory period: 2 ms

## Citation

If you use this concept or code in your research or project, please cite:

```
Spherical Brain Space Simulator
Author: rubyguyan
GitHub: https://github.com/rubyguyan/spherical-brain-space
Release Date: March 2026
```

## License

- **Code**: MIT License - Free to use, please credit the source
- **Concept Description**: CC BY 4.0 - Please credit the source when reposting

## Acknowledgments

- [FlyWire](https://flywire.ai/) - Fruit fly whole-brain connectome project
- [navis](https://github.com/navis-org/navis) - Neuron morphology analysis tool
