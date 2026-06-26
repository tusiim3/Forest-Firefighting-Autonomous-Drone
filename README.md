# Forest-Firefighting-Autonomous-Drone

# Forest Firefighting Autonomous Drone

[cite_start]**Student:** Tusiime Mark [cite: 28]  

[cite_start]**Course:** CSC 2207 Robotics [cite: 28]  

## Project Overview

[cite_start]This repository contains the autonomous control logic for a Mavic 2 Pro drone tasked with detecting and extinguishing forest fires within a dynamic simulation[cite: 30]. [cite_start]The system executes locomotion, visual perception, and spatial navigation[cite: 30].

## Evaluation Deliverables Guide

### 1. Controller Source Code

[cite_start]The core operational scripts are located within the `controllers/` directory[cite: 30]:

 *[cite_start]*`controllers/my_mavic/my_mavic.py`**: Contains the flight control loop, the boustrophedon grid navigation protocol, and the state machine[cite: 28].

 *[cite_start]*`controllers/my_mavic/fire_detector.py`**: Contains the camera-based perception module, utilising HSV colour segmentation to isolate smoke signatures[cite: 28].

### 2. Technical Report

[cite_start]The project report is formatted as a PDF[cite: 30]. It contains:

* [cite_start]The system architecture diagram[cite: 30].

* [cite_start]Key algorithmic implementations[cite: 30].

* [cite_start]Development challenges and applied solutions[cite: 30].

* [cite_start]Empirical performance results, including total extinguishment time[cite: 30].

### 3. Live Demonstration

[cite_start]The live simulation deployment and code explanation constitute the final deliverable[cite: 30].

## Execution Environment

The operational cycle requires the following host specifications:

 *[cite_start]**Simulator:** Webots R2021b[cite: 29].

 *[cite_start]**Runtime:** Python 3.9.25[cite: 29].

 *[cite_start]**Perception Libraries:** OpenCV 4.13.0.92 and NumPy 2.0.2[cite: 29].