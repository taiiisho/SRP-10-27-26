# Prompt for Claude Opus 5.5: Raspberry Pi hand-trajectory prediction research code

Copy everything below the line into a new Claude Code session (ideally one running on the Raspberry Pi, or in an empty repo you will copy to the Pi).

---

<role>
You are a senior robotics/ML engineer and research-software mentor. You are building the complete software for a high-school science research project (two 11th-graders, category: Computer Science). The code must be correct enough to publish results from, and readable enough that two students can understand, run, and defend every part of it at a science fair.
</role>

<project>
**Title:** Physics-based prediction versus edge AI prediction

**Research question:** How does a physics-based model compare to edge AI in predicting hand movements?

**Hypothesis:** If the prediction horizon increases, then the ML model will eventually produce lower trajectory error than the physics-based model.

**Variables**
- Independent: prediction time horizon. Required: 50 ms, 100 ms, 200 ms. Also evaluate 150, 300, 400 and 500 ms so we can see where (and whether) the curves cross. The word "eventually" in the hypothesis means we need horizons long enough to see a crossover.
- Dependent: hand trajectory prediction error, i.e. Euclidean distance between the predicted and the actually measured hand position at t + horizon.
- Controlled: recording environment, camera angle/position, frame rate, camera resolution, the AI accelerator, and the single hand landmark used for all measurements.

**Hardware:** Raspberry Pi (assume Pi 5) + Raspberry Pi AI HAT+ 2 (believed to be Hailo-10H based; verify against current official Raspberry Pi / Hailo documentation, do not rely on memory for APIs), a camera (assume Pi Camera Module 3 but make it configurable; a USB webcam must also work), camera mount ~1 m from the participant's hand, measuring tape, storage device.

**Procedure we must support**
1. Camera is fixed ~1 m from the hand. Orientation, height, distance, resolution and frame rate stay constant.
2. A hand-landmark detection model finds one consistent point on the hand every frame.
3. Software records hand position + timestamp for every frame.
4. A physics-based prediction model and an ML prediction model are built.
5. Testing: record a series of hand movements. Each movement starts from roughly the same start region and moves toward a predetermined target location. Record the trajectory. For each trajectory, both models predict hand position at multiple horizons; every prediction is compared with the actually measured position; error is the Euclidean distance.
6. At least 30 independent movement trials, at different speeds and directions. The ML model is evaluated only on unseen data.

**Data the system must record (per frame):** timestamp, hand x, hand y, estimated velocity, estimated acceleration, prediction horizon.
**Data the system must record (per prediction):** physics prediction, ML prediction, actual hand position, physics error, ML error.

**Physics models (from the proposal's background research)**
- *Minimum-Jerk model* (Flash & Hogan, 1985): the nervous system minimises the integral of squared jerk. The Euler–Lagrange solution is a fifth-order polynomial: `x(t) = x0 + (xf − x0) · (10τ³ − 15τ⁴ + 6τ⁵)`, `τ = (t − t0)/T`, fully determined by start point, end point and movement duration. It is a kinematic model: it ignores muscles, mass and forces.
- *Newton–Euler (Euler integration) model*: step the state forward in small time steps: `v[n+1] = v[n] + a[n]·Δt`, `x[n+1] = x[n] + v[n]·Δt`.
- The proposal says the two "work hand in hand": Minimum-Jerk supplies the shape of the motion; Euler stepping rolls the state forward.

**ML models (from the proposal's background research):** learn a mapping from past hand positions to future positions by minimising prediction error on training data. Candidates named in the proposal: recurrent networks (LSTM) and Gaussian process regression. Key risk: overfitting / memorising training data, so held-out evaluation is mandatory. Reference for the ML side: Gamage et al. (2021), "So Predictable! Continuous 3D Hand Trajectory Prediction in Virtual Reality", UIST '21.
</project>

<assumptions>
Do not stop to ask about these. Adopt them, state them in the README, and make each one a config value so it is easy to change.
- Pi 5, Raspberry Pi OS (Bookworm or newer), Python 3.11+, `picamera2` for the Pi camera, OpenCV for USB cameras and video files.
- 2D tracking in image coordinates is the primary data (x, y in pixels, converted to centimetres via calibration). The proposal's records list only x and y. If the detector provides a relative z, log it as an extra column but do not use it in the headline comparison.
- Default tracked landmark: the middle-finger MCP knuckle (MediaPipe landmark 9). It is more stable than a fingertip when fingers flex. Configurable, but fixed for the whole study.
- Default camera settings: 60 fps or higher if the sensor, detector and Pi can sustain it without dropped frames (30 fps is too coarse for a 50 ms horizon: that is 1.5 frames). Fixed exposure, fixed white balance, and manual focus at ~1 m (Camera Module 3: lens position ≈ 1.0 dioptre). All settings are logged in session metadata.
- One tracked hand, one participant at a time. Participants are identified only by a pseudonymous ID.
- Privacy: by default store only landmark coordinates, never video. Saving video requires an explicit `--save-video` flag.
</assumptions>

<deliverables>
Build a Python project with this structure (adjust names if you have a good reason, but keep the separation of concerns):

```
README.md                  plain-language guide: setup, wiring, how to run each step, how to collect data
pyproject.toml             pinned/minimum versions
configs/default.yaml       every tunable value: camera, landmark id, horizons, filter params, model hyperparameters, seeds
handpredict/
  capture/                 camera sources: Picamera2, OpenCV USB, video-file replay, synthetic
  landmarks/               LandmarkDetector interface + implementations (see 1)
  calibration.py           pixels → centimetres
  recording.py             trial recorder + session metadata
  kinematics.py            causal filtering, velocity/acceleration, onset/offset segmentation
  physics/                 constant-velocity, Euler constant-acceleration, minimum-jerk (closed form and Euler-rolled)
  ml/                      dataset builder, models, training, ONNX export, inference wrapper
  evaluation/              prediction table, metrics, statistics, plots, auto-generated summary
  live_demo.py             real-time overlay of both predictions
scripts/ (or CLI subcommands)  calibrate, record, build-dataset, train, evaluate, demo, benchmark
tests/                     pytest suite
```

### 1. Capture and landmark detection
- Define a `LandmarkDetector` interface and implement:
  1. `MediaPipeDetector` (CPU): must work on any machine so the students can develop on a laptop.
  2. `HailoDetector`: runs a hand-landmark model on the AI HAT+ 2. Research the current official way to do this (Hailo model zoo / `hailo-apps` / `hailo-rpi5-examples` / Raspberry Pi `rpicam-apps` post-processing / HailoRT Python API) and use it. If you are unsure of an API, isolate it behind the interface, mark the uncertain lines with `# VERIFY ON HARDWARE`, and say so in your final report. Never invent function names silently.
- Sources: Picamera2, USB webcam, video file (for replaying recorded sessions deterministically), and a synthetic source that generates minimum-jerk hand paths with noise (for tests/CI; synthetic data must be clearly labelled as such and never mixed into real results).
- Timestamps: use the camera's sensor timestamp when available, otherwise a monotonic clock taken at frame capture. Log detector latency separately. Never use wall-clock time. Count dropped frames and missed detections.
- Missing detections: interpolate gaps of ≤ 3 frames and flag them; discard trials that exceed a configurable bad-frame fraction.

### 2. Calibration
- A `calibrate` tool: the user places the measuring tape in the measurement plane, clicks two points in the image (or enters pixel coordinates) at a known distance, and the tool stores cm-per-pixel plus the start region and target location (pixel coordinates) in the session file. Error is reported in both pixels and centimetres.
- Include an option to mark the start region and target by clicking.

### 3. Trial recording
- CLI recorder: operator presses Enter → short countdown → trial recorded for a fixed duration (default 2.5 s) or until the hand has stopped at the target. Operator labels each trial with `participant_id`, `speed_condition` (slow/medium/fast), `direction` or `target_id`, and free-text notes. Must work over SSH without a display (preview optional).
- Each session writes: `session.json` (all camera settings, resolution, fps, landmark id, calibration, software versions, git hash, config snapshot, detector backend, hardware description) and one raw file per trial (`trial_XXX.csv`: frame index, timestamp, x_px, y_px, x_cm, y_cm, detection confidence, interpolated flag).
- Raw data is never overwritten and never edited by later steps.
- Support recording ≥ 30 trials per session and multiple sessions/participants; the dataset builder merges them.

### 4. Kinematics
- Causal filtering only (e.g. a One Euro filter, a Kalman filter with a constant-acceleration model, or a causal Savitzky–Golay variant). Every predictor may use only samples at or before the prediction time. A non-causal smoother is a data leak and is forbidden anywhere in the prediction path.
- Estimate velocity and acceleration from the causal-filtered signal with the actual (non-uniform) frame Δt. Record them per frame as the proposal requires.
- Segment each trial into movement onset and offset using a speed threshold with hysteresis (configurable). Plot the segmentation for visual checking.
- **Ground truth** for horizon h at time t is the measured landmark position at t + h, obtained by linear interpolation of the raw (unfiltered) measured positions. Document that detector noise therefore sets a floor on error that is common to both models.

### 5. Physics models
All physics predictors share one interface: `predict(history, horizons_ms) → positions`, where `history` contains only past samples.
- `constant_velocity` and `euler_constant_acceleration` (the proposal's Newton–Euler equations, stepped with a small internal Δt; the acceleration may optionally decay, behind a config flag, but the default is the proposal's constant `a`). These are baselines/ablations.
- **`minimum_jerk` (the primary physics model).** The target is predetermined and known (that is part of the experimental design), so the unknowns are onset time t0 and duration T (and optionally a "free-target" variant that also fits xf, as an ablation). Fit them online by bounded nonlinear least squares to the observed samples since onset, extrapolate along the straight line from start to target, hold at the target once τ ≥ 1, and fall back to constant-velocity (and log that it fell back) when the fit is unavailable or ill-conditioned (e.g. too few samples after onset). Provide both the closed-form polynomial and an Euler-rolled version (use the analytic acceleration profile and step v and x with the Newton–Euler update). The two must agree within a tiny tolerance; that is a unit test.
- The headline "physics-based model" in all results is `minimum_jerk` (Euler-rolled). The other physics variants appear as extra rows/curves, not as cherry-picked alternatives. Decide this before looking at test results and state it in the README.

### 6. ML models
- Dataset builder: sliding windows over each trial. Input = last W samples (default ≈ 150–200 ms of history, configurable) of positions relative to the current position, plus the current absolute position (ablation flag to remove it), plus per-sample Δt. Output = displacement from the current position at every horizon simultaneously (multi-output head).
- Models: (a) a small GRU or LSTM (primary), (b) a small MLP or 1-D CNN (accelerator-friendly), (c) optional Gaussian process regression baseline via scikit-learn. Train in PyTorch.
- Loss: Euclidean distance (L2 norm of the error vector) or a smooth variant, so training matches the evaluation metric.
- **Splits are by trial, never by frame.** Frames within a trial are highly autocorrelated, so a random frame split leaks the answer. Use a held-out test set of whole trials (and, if multiple participants exist, ideally whole participants) that is never touched during model/hyperparameter selection; use grouped k-fold cross-validation by trial for the model-selection and for reporting variance when the dataset is small (~30 trials). Fix random seeds, train multiple seeds, report mean ± sd.
- Light augmentation is allowed on the training split only (small noise, mild speed scaling); document it.
- Export the chosen model to ONNX, run it on the Pi with ONNX Runtime, and check that its outputs match PyTorch.
- **Edge-AI deployment:** the research question is about *edge* AI, so make a real attempt to run the ML predictor on the AI HAT+ 2. Research current Hailo toolchain constraints (the Hailo Dataflow Compiler runs on an x86 Linux machine and not on the Pi; recurrent layers have limited support, which is why the MLP/CNN variant exists). Provide a documented `export_hailo` path as a best-effort stretch goal. The default, always-working path is ONNX Runtime on the Pi CPU for the predictor. Record in session metadata and in the results which hardware ran (a) landmark detection and (b) the ML predictor, and benchmark both. Never claim the predictor ran on the accelerator unless it did.

### 7. Evaluation and statistics
- A single evaluation pipeline produces a tidy `predictions.csv` with, per (trial, frame, horizon, model): `trial_id, participant_id, speed_condition, split, timestamp, x, y, vx, vy, ax, ay, horizon_ms, physics_pred_x/y, ml_pred_x/y, actual_x/y, physics_error, ml_error` (plus columns for the extra baselines, including a zero-motion "stay where you are" baseline as the sanity floor). Errors in both pixels and cm.
- All models are evaluated on exactly the same frames, horizons and ground truth. Primary analysis window: frames inside the detected movement [onset, offset] where t + h ≤ offset. Secondary: whole trial.
- Aggregate to **trial level** before statistics (mean error per trial per model per horizon), because frames are not independent. Then:
  - mean, median, RMSE, 95th percentile per model per horizon, with bootstrap 95% CIs resampled over trials;
  - paired comparison of physics vs ML per horizon (Wilcoxon signed-rank on per-trial paired errors, plus paired effect size), with Holm correction across horizons;
  - estimate the crossover horizon (where ML − physics error changes sign) with a bootstrap CI, or report that no crossover was observed in the tested range;
  - breakdowns by speed condition and direction;
  - results are reported for the held-out test set only; training-set error is shown only as a labelled overfitting diagnostic.
- Plots (PNG + CSV of plotted values): error vs horizon with CIs for every model; per-trial paired differences; example trajectories showing actual vs both predictions at 50/100/200 ms; error vs hand speed; learning curves. Colour-blind-safe palette, labelled axes with units.
- `evaluate` also writes an auto-generated `results/summary.md` containing the numbers, the sample size (number of trials, participants, frames), and a plain statement of whether the data support, contradict, or fail to resolve the hypothesis. It must not overclaim; with ~30 trials it should say so.

### 8. Edge-performance benchmark
- `benchmark` command: end-to-end FPS, per-stage latency (capture, landmark detection, filtering, physics predict, ML predict), CPU load/temperature, dropped frames, for each backend. Make clear in the output whether the predictor's latency is smaller than the shortest horizon (50 ms), since a prediction that arrives later than its horizon is useless in a real-time system, which is a strong point for the write-up.

### 9. Live demo
- `demo` command: live camera feed with the tracked landmark, plus the physics and ML predicted positions drawn at a selectable horizon, and a rolling error readout. Works headless (writes frames or just logs) and with a display.
</deliverables>

<scientific_validity_rules>
These are non-negotiable. If you find yourself violating one, stop and fix it.
1. No future information in any prediction: causal filters only; features at time t use samples ≤ t.
2. No train/test leakage: split by whole trials (or participants); test set untouched until the final evaluation; normalisation statistics computed on training data only.
3. Both models see the same history, same sampling, same ground truth, same frames. The physics model gets the known target because the design gives it to both; the ML model's access to the target must match (absolute position input / target id), and this is documented as a design choice.
4. Controlled variables are logged in every session file and the evaluator warns if sessions being merged differ in resolution, fps, camera pose, landmark id or backend.
5. No fabricated data. Synthetic data is allowed only for tests, demos and `--selftest`, is labelled `synthetic` in the file and the filename, and the evaluator refuses to mix it with real data unless explicitly told to.
6. The summary reports what the data show, including null or opposite results.
7. Everything reproducible from `session.json` + config + seeds.
</scientific_validity_rules>

<quality_bar>
- Tests (pytest) that run on a laptop with no hardware: minimum-jerk closed form vs Euler-rolled agreement; minimum-jerk predictor recovers a noiseless synthetic trajectory with near-zero error; constant-velocity is exact on constant-velocity motion; causal filter never uses future samples (perturb a future sample and assert earlier outputs are unchanged); ground-truth interpolation; trial-level split has no overlap; metrics on hand-computed cases; dataset builder window shapes; ONNX output equals PyTorch output; the full pipeline runs end to end on synthetic data in under a few minutes.
- Type hints, docstrings that explain the *why* in plain English, small functions, no magic numbers (everything in config), logging instead of prints, clear error messages when hardware is missing ("Hailo device not found; falling back to CPU detector").
- Comment density: match a clear teaching style. Explain the physics and statistics concepts where they appear, but don't narrate trivial lines.
- README for non-experts: hardware assembly checklist, installing the AI HAT+ 2 software stack, camera set-up and the measurement-area layout, how to run a data-collection session (with a protocol checklist: lighting, background, hand start/target markers, trial counts per speed/direction, rest breaks), how to train, how to evaluate, how to interpret the plots, and a troubleshooting section.
- Keep dependencies modest and pinned; the base install must work on a laptop without Hailo packages.
</quality_bar>

<process>
1. Before writing code, read the repo and the environment (`uname -a`, Python version, whether `picamera2`, `hailo` packages, and a camera are present). Adapt to what is actually there.
2. Post a short plan: architecture, the milestone order, and any assumptions you are changing from the list above and why. Then proceed without waiting for approval.
3. Build in milestones, running the tests at each one: (M1) project skeleton + synthetic source + kinematics + tests; (M2) physics models + tests; (M3) recording/calibration + MediaPipe detector + video replay; (M4) dataset builder + ML training + ONNX; (M5) evaluation/statistics/plots on synthetic data; (M6) Hailo detector, benchmark, live demo, README. Commit after each milestone with a clear message.
4. Use the web or official docs for Hailo / Picamera2 / MediaPipe API details when you can. When you cannot verify something on real hardware, say so explicitly rather than guessing.
5. Verify your work: run the full test suite and the end-to-end synthetic pipeline, and show the output. Do not claim anything works that you did not run.
6. Final report (concise): what was built, how to run each step, what was verified vs. what still needs verification on the actual Pi + AI HAT+ 2, every assumption made, and any risks to the science (e.g. few trials, single participant, frame rate vs 50 ms horizon, the predictor not running on the accelerator).

Ask me a question only if you are truly blocked on something that cannot be resolved by a reasonable default.
</process>

<out_of_scope>
Whole-body or skeleton tracking, 3D reconstruction with multiple cameras, cloud services, a web UI, and anything that sends participant data off the device.
</out_of_scope>
