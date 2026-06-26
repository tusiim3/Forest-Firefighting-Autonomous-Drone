# Forest Firefighting Autonomous Drone

**Student:** Tusiime Mark  
**Registration Number:** 22/U/11684/PS   
**Student Number:** 2200711684  

## Project Overview

This repository contains the autonomous control logic for a Mavic 2 Pro drone tasked with detecting and extinguishing forest fires within a dynamic simulation. The system executes locomotion, visual perception, and spatial navigation.

## Evaluation Deliverables Guide

### 1. Controller Source Code

The core operational scripts are located within the `controllers/` directory:

- `**controllers/my_mavic/my_mavic.py`**: Contains the flight control loop, the boustrophedon grid navigation protocol, and the state machine.
- `**controllers/my_mavic/fire_detector.py**`: Contains the camera-based perception module, utilising HSV colour segmentation to isolate smoke signatures.

### 2. Technical Report

The project report is formatted as a PDF. It contains:

- The system architecture diagram.
- Key algorithmic implementations.
- Development challenges and applied solutions.
- Empirical performance results, including total extinguishment time.

## Execution Environment

The operational cycle requires the following host specifications:

- **Simulator:** Webots R2021b.
- **Runtime:** Python 3.9.25.
- **Perception Libraries:** OpenCV 4.13.0.92 and NumPy 2.0.2.

