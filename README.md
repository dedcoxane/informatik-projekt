# informatik-projekt
Data-driven risk assessment for lane changing using HMM and classical safety models, achieving 93.8% crash detection recall on highway trajectories.
Lane Changing Management / Spurwechselmanagement
This repository contains the code and documentation for the Informatik Projekt at Frankfurt University of Applied Sciences, focusing on data-driven risk assessment for lane changing maneuvers on highways.

Overview
The project implements and evaluates multiple safety models for autonomous driving and advanced driver assistance systems (ADAS), with a particular focus on cut-in, cut-out, and car-following scenarios as defined by UNECE Regulation R157. The work combines classical deterministic models with probabilistic machine learning approaches, all validated using the highD drone dataset of naturalistic vehicle trajectories on German highways.

# Key Features
Data Pipeline: Complete preprocessing pipeline for the highD dataset including filtering, interpolation, resampling (40ms), and synchronization of trajectory data

Feature Engineering: Extraction of dynamic safety features including Time-to-Collision (TTC), TTC derivative, Distance Headway (DHW), Time Headway (THW), and lateral motion intensity

Driver Classification: Quantile-based classification of driving styles into aggressive, normal, and defensive categories

Multiple Safety Models:

HMM (Hidden Markov Model) with 4 hidden states for probabilistic risk assessment

FSM (Fuzzy Safety Model) with CFS and PFS metrics

RSS (Responsibility-Sensitive Safety) with formal safety distances

Reg157 (UNECE R157 compliant model)

CC_human_driver (human-like car-following behavior)

Threshold Optimization: Systematic optimization of decision thresholds for optimal F1-score

Comprehensive Evaluation: Accuracy, Precision, Recall, F1-Score, confusion matrices, and parameter studies

# Results
The HMM-based approach achieved the best performance across all scenarios:

Cut-in crash rate: 10.2% (compared to 22.1% for human-like model)

Car-following crash rate: 6.4%

Recall after optimization: 93.8% (15/16 crash sequences detected)

Optimal F1-score: 0.40

Repository Structure
data_processing/ - Data loading, filtering, and preprocessing scripts

feature_engineering/ - Feature extraction and driver classification

models/ - Implementation of all five safety models

evaluation/ - Metrics calculation, threshold optimization, and visualization

notebooks/ - Jupyter notebooks for exploratory analysis and results visualization

config/ - Configuration files and hyperparameters

Technologies Used
Python 3.12

NumPy, pandas for data processing

hmmlearn for Hidden Markov Models

scikit-learn for metrics and preprocessing

Matplotlib, seaborn for visualization

Jupyter notebooks for interactive analysis

References
This project builds upon the highD dataset [1] and implements models based on RSS [6], FSM [2], and UNECE R157 [8] specifications. The complete methodology and results are documented in the accompanying thesis.


Author: Mohammad Ahmad Khanm
Supervisor: Prof. Jamal Rayan
Institution: Frankfurt University of Applied Sciences
Semester: Winter 2025/2026
