# AutoRNet: Automatically Optimizing Heuristics for Robust Network Design via LLMs

## Overview

AutoRNet is a framework that combines Large Language Models (LLMs) and Evolutionary Algorithms (EAs) to automatically generate robust heuristics for network optimization. It focuses on enhancing network robustness by leveraging domain-specific knowledge and adaptive strategies. This repository includes the implementation of AutoRNet, along with example datasets and tools for evaluating network robustness.

### Key features:
	•	Integration of LLMs with EAs for automated heuristic design.
	•	Network Optimization Strategies (NOS) for domain-specific problem solving.
	•	Adaptive Fitness Function (AFF) to balance convergence and diversity during optimization.

### Getting Started

Prerequisites

	•	Python 3.8+
	•	Required libraries (install using pip)
 
### Outputs

	•	Optimized networks: Results are saved in the output folder or displayed in the visualization tool.
	•	Robustness scores: Evaluations are logged during each run.

### Key Modules

	•	problem/: Defines network structures and constraints for optimization.
	•	genetic.py: Implements the core EA logic, including variation and selection operations.
	•	prompt.py: Generates problem-specific prompts for LLMs.
	•	connector.py: Manages the interaction between LLMs and EAs.
	•	viewer/: Provides visualization utilities for analyzing and presenting results.



### Citation
If you use AutoRNet in your research, please cite our paper:

 > @article{HeYu2024,
  title={AutoRNet: Automatically Optimizing Heuristics for Robust Network Design via Large Language Models},
  author={He Yu and Jing Liu},
  journal={Preprint},
  year={2024}
}
