"""handpredict: physics-based versus edge-AI prediction of hand movements.

The package is organised by the steps of the experiment:

capture/       cameras and replay sources (Picamera2, USB webcam, video file, synthetic)
landmarks/     hand-landmark detectors (MediaPipe on the CPU, Hailo on the AI HAT+ 2)
calibration    pixels -> centimetres, start region and target locations
recording      trial recorder and session metadata
kinematics     causal filtering, velocity, acceleration, movement segmentation
physics/       physics-based predictors (constant velocity, Euler, minimum jerk)
ml/            machine-learning predictors (dataset, training, ONNX export)
evaluation/    prediction table, metrics, statistics, plots and summary
"""

__version__ = "0.1.0"
