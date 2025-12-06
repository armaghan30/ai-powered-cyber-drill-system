# 🚀 AI-Powered Cyber Drill System  
### Reinforcement Learning-based Red & Blue Teams on a Custom Cyber Range Environment  

This project implements a **fully interactive cyber drill system** powered by **Reinforcement Learning (RL)**.  
It simulates realistic cybersecurity scenarios where:

- A **Red Agent (attacker)** uses Deep Q-Learning (DQN)  
- A **Blue Agent (defender)** uses rule-based logic or RL (MARL version)  
- Both operate inside a **custom cyber range environment** with configurable network topologies  
- Simulations produce detailed logs, rewards, attack chains, and performance metrics  

This system is designed as part of a **Final Year Project (FYP)** and aims to model cyber attack and defense behaviors that adapt over time.

---

# 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Environment Design](#environment-design)
- [Red Agent (DQN)](#red-agent-dqn)
- [Blue Agent](#blue-agent)
- [Multi-Agent RL (MARL)](#multi-agent-rl-marl)
- [Experimental Component: Tiny Agent](#experimental-component-tiny-agent)
- [Project Folder Structure](#project-folder-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Simulation Logging](#simulation-logging)
- [Configuration (Topologies & Scenarios)](#configuration-topologies--scenarios)
- [Results & Visualizations](#results--visualizations)
- [Dashboard (Planned Work)](#dashboard-planned-work)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

---

# 🔍 Overview

The AI-Powered Cyber Drill System models a simplified enterprise network where agents interact turn-by-turn.  

**Red Agent:**  
- Performs scanning  
- Identifies vulnerabilities  
- Attempts exploits  
- Moves laterally  
- Tries to compromise all critical hosts  

**Blue Agent:**  
- Detects malicious activity  
- Chooses defensive responses: isolate, patch, restore  
- Uses rule-based logic or RL depending on the configuration  

The system includes:  
✔ A custom RL environment (Gymnasium-style)  
✔ Deep Reinforcement Learning (DQN) implementation  
✔ Multi-agent RL (MARL) extension  
✔ Configurable YAML-based topology definitions  
✔ Full logging of attack/defense sequences  
✔ Reward shaping engine  
✔ Visualizations of training and evaluation  

---

# ✨ Key Features

### 🔥 Custom Cyber Range Environment
- Built from scratch (not using NASim or CyberBattleSim)
- Supports arbitrary network topologies via YAML files
- Tracks:
  - vulnerabilities  
  - services  
  - host status  
  - compromise state  
  - red and blue actions  

---

### 🤖 Red Agent – Deep Q-Learning (DQN)
- Fully implemented using PyTorch  
- Epsilon-greedy exploration  
- Experience replay buffer  
- Target network updates  
- Supports greedy evaluation  

---

### 🛡️ Blue Agent
Two modes:

#### 1️⃣ Rule-Based  
- Simple deterministic defense  
- Picks actions such as isolate, patch, restore  
- Reacts to compromised hosts and scans  

#### 2️⃣ RL-Based (MARL)  
- Works jointly with Red Agent in a multi-agent setting  
- DQN for both attacker and defender  
- Cooperative/competitive dynamics  

---

### 👥 Multi-Agent RL (MARL)
Includes:

- `train_marl_dqn.py`  
- `eval_marl_dqn.py`  
- Shared environment for simultaneous Red & Blue actions  
- Separate DQNs for each agent  
- Joint reward shaping  

Models included:  
- `marl_red_dqn.pth`  
- `marl_blue_dqn.pth`

---

# 🧠 Environment Design

The environment is defined in:

orchestrator/rl_env_red.py
orchestrator/rl_env_blue.py
orchestrator/multi_agent_env.py


It supports:

- Gymnasium-style API  
  - `reset() → (obs, info)`  
  - `step(action) → (obs, reward, terminated, truncated, info)`  
- Vulnerability discovery  
- Exploitation outcomes  
- Defensive responses  
- Episode termination based on:
  - max steps  
  - all hosts compromised  
  - blue containment success  

A reward engine assigns:

- Positive reward for successful exploits  
- Negative reward for failed actions  
- Penalties for wasted steps  
- Bonus for full network compromise  

---

# 🔺 Red Agent (DQN)

Implemented in:

orchestrator/dqn_agent_red.py


Supports:

- Replay buffer  
- Target network  
- Neural network policy  
- Epsilon decay  
- Greedy evaluation  

Training script:

orchestrator/train_dqn_red.py


Evaluation script:

orchestrator/eval_report_red.py


Produces:

- reward plots  
- compromised host plots  
- simulation logs  

---

# 🔵 Blue Agent

Two implementations:

### Rule-Based:
orchestrator/agents/blue_agent.py


### RL-Based (MARL):
orchestrator/train_marl_dqn.py
orchestrator/eval_marl_dqn.py


---

# 🧪 Experimental Component: Tiny Agent

The `tiny_agent/` folder contains an **early prototype environment** used to test:

- RL loops  
- Action/state formatting  
- DQN implementation  
- Observation handling  

This module is *not part of the final system*, but it demonstrates the evolution of the project and is included for transparency.

---

# 📁 Project Folder Structure

AI-Powered Cyber Drill System/
│
├── orchestrator/
│ ├── agents/
│ ├── tests/ (train_dqn_red.py, eval_report_red.py, train_marl_dqn.py, etc.)
│ ├── rl_env_red.py
│ ├── rl_env_blue.py
│ ├── multi_agent_env.py
│ ├── reward_engine.py
│ ├── orchestrator_core.py
│ └── sample_topology.yaml
│
├── configs/
│ ├── sample_topology.yaml
│ └── scenario_definitions.yaml
│
├── tiny_agent/
│ ├── tiny_env.py
│ ├── tiny_state.py
│ ├── tiny_actions.py
│ ├── train_tiny.py
│ └── eval_tiny.py
│
├── dashboard/ (planned, not yet implemented)
├── docs/
├── red_dqn_model.pth
├── blue_dqn_model.pth
├── marl_red_dqn.pth
├── marl_blue_dqn.pth
├── *.csv (training logs)
├── *.png (plots)
└── simulation_log.


---

# 📊 Training

### Train Red DQN:
```bash
python -m orchestrator.train_dqn_red



