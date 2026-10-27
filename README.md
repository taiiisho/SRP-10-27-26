# Physics-based prediction versus edge-AI prediction of hand movements

**Research question:** How does a physics-based model compare to edge AI in predicting hand movements?

**Hypothesis:** If the prediction horizon increases, then the ML model will eventually produce lower trajectory
error than the physics-based model.

This repository is the complete software for the study: it records hand movements with a Raspberry Pi 5,
a Camera Module 3 and the Raspberry Pi AI HAT+ 2 (Hailo-10H accelerator), predicts where the hand will be
50-500 ms later with a physics model (minimum jerk) and with machine-learning models (GRU, MLP, Gaussian
process), and compares them with proper statistics.

```
camera ──> hand-landmark detector ──> trial recorder ──> build-dataset ──> train ──> evaluate ──> summary.md
(60 fps)   (MediaPipe on the CPU,     (raw files,        (causal filter,   (ML on     (same frames,
            or Hailo-10H)              never edited)       segmentation,     training   trial-level
                                                           train/test split) trials)   statistics)
```

---

## Contents

1. [What is in this repository](#1-what-is-in-this-repository)
2. [Quick start on a laptop (no camera needed)](#2-quick-start-on-a-laptop-no-camera-needed)
3. [Hardware checklist and assembly](#3-hardware-checklist-and-assembly)
4. [Installing the software on the Raspberry Pi](#4-installing-the-software-on-the-raspberry-pi)
5. [Setting up the measurement area](#5-setting-up-the-measurement-area)
6. [Calibration](#6-calibration)
7. [Recording a data-collection session](#7-recording-a-data-collection-session)
8. [From recordings to results](#8-from-recordings-to-results)
9. [How to read the results](#9-how-to-read-the-results)
10. [Edge AI on the AI HAT+ 2](#10-edge-ai-on-the-ai-hat-2)
11. [Live demo](#11-live-demo)
12. [Design decisions and assumptions](#12-design-decisions-and-assumptions)
13. [Troubleshooting](#13-troubleshooting)
14. [For developers: tests, reproducibility, code map](#14-for-developers-tests-reproducibility-code-map)
15. [References](#15-references)

---

## 1. What is in this repository

| Path | What it is |
|---|---|
| `configs/default.yaml` | **Every** tunable number (camera, landmark, filter, thresholds, horizons, models, statistics, seeds), each with a comment. |
| `handpredict/capture/` | Frame sources: Raspberry Pi camera (Picamera2), USB webcam (OpenCV), video replay, synthetic. |
| `handpredict/landmarks/` | Hand-landmark detectors: MediaPipe (CPU), Hailo-10H (AI HAT+ 2), synthetic (tests). |
| `handpredict/calibration.py` | Pixels to centimetres, start region, targets. |
| `handpredict/recording.py`, `record_session.py` | Trial recorder, session files, operator prompts, replay. |
| `handpredict/kinematics.py` | Causal filtering, velocity, acceleration, movement onset/offset, ground truth. |
| `handpredict/physics/` | Zero motion, constant velocity, Newton-Euler constant acceleration, minimum jerk (closed form, Euler-rolled, re-planned, free target). |
| `handpredict/ml/` | Training windows, GRU/LSTM, MLP, Gaussian process, training with cross-validation, ONNX export, Hailo export. |
| `handpredict/dataset.py`, `splits.py` | `build-dataset`: merging sessions, quality checks, train/test split by whole trials. |
| `handpredict/evaluation/` | Prediction table, metrics, statistics, plots, `summary.md`. |
| `handpredict/online.py`, `live_demo.py`, `benchmark.py` | Real-time use: live demo and edge benchmark. |
| `tests/` | 100+ automatic tests (`pytest`). |
| `scripts/setup_pi.sh` | Installs everything on the Pi. |

All commands are sub-commands of one program, `handpredict` (run `handpredict --help`):

| Step | Command |
|---|---|
| Check the Pi, camera and accelerator | `handpredict check-hardware` |
| Download detector models | `handpredict download-models` |
| Calibrate | `handpredict calibrate ...` |
| Record trials | `handpredict record ...` |
| Re-run a detector on saved videos | `handpredict replay ...` |
| Build the dataset | `handpredict build-dataset` |
| Train the ML models | `handpredict train` |
| Evaluate (the result) | `handpredict evaluate` |
| Live demo | `handpredict demo ...` |
| Speed on the edge | `handpredict benchmark ...` |
| ML model for the Hailo-10H | `handpredict export-hailo ...` |
| Whole pipeline on synthetic data | `handpredict selftest` |

Every command accepts `--config my_settings.yaml` (only the keys you change) and `--set section.key=value`,
e.g. `--set camera.fps=90`. Unknown keys are rejected, so a typo cannot silently do nothing.

---

## 2. Quick start on a laptop (no camera needed)

Python 3.11 or newer. On Windows use WSL or adapt the commands.

```bash
git clone <this repository> && cd <repository>
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[mediapipe,ml,dev]"
handpredict download-models mediapipe test-assets   # MediaPipe model + a test photo
handpredict selftest                                 # ~2-3 minutes: record -> build -> train -> evaluate (SYNTHETIC)
pytest                                               # ~2 minutes: all tests
```

`selftest` prints a `summary.md` with a big **SYNTHETIC DATA** warning. Synthetic data only proves the
software works; it says nothing about real hands and is never mixed with real data (the software refuses).

---

## 3. Hardware checklist and assembly

- Raspberry Pi 5 (8 GB recommended) with the **official 27 W USB-C power supply** and an **active cooler**.
- Raspberry Pi **AI HAT+ 2** (Hailo-10H, 40 TOPS INT4, 8 GB on-board memory) with its heatsink, spacers, GPIO stacking header and PCIe ribbon.
- **Camera Module 3** (standard lens, not the wide one) and a **Pi 5 camera cable** (Pi 5 uses the smaller 22-pin connector).
- A sturdy tripod or clamp that holds the camera **1 m** from the plane the hand moves in.
- Measuring tape (for calibration), masking tape or stickers (start and target markers), a plain matte background cloth.
- Two bright LED lamps (flicker-free); a microSD card (32 GB+) and a **USB stick or SSD for backups**.

Assembly (follow Raspberry Pi's official AI HAT+ documentation for the details; Pi unplugged throughout):

1. Fit the Active Cooler to the Pi 5, and the heatsink to the AI HAT+ 2.
2. Connect the camera cable to the Pi 5 **before** mounting the HAT (the official guide recommends this order)
   and route it so the HAT will not pinch it.
3. Mount the AI HAT+ 2: spacers, GPIO stacking header, PCIe ribbon into the Pi (contacts towards the USB
   ports), HAT onto the spacers, ribbon into the HAT.
4. Only then connect power.

---

## 4. Installing the software on the Raspberry Pi

Use **Raspberry Pi OS (64-bit) Trixie**: the official documentation requires it for the AI HAT+ 2. The
script below does everything; the individual steps are listed for understanding and debugging.

```bash
git clone <this repository> && cd <repository>
bash scripts/setup_pi.sh      # then: sudo reboot   (and run it again if it asked you to reboot early)
```

What it does:

```bash
sudo apt update && sudo apt full-upgrade -y && sudo rpi-eeprom-update -a
sudo apt install -y dkms hailo-h10-all        # AI HAT+ 2 driver + HailoRT + PyHailoRT. NOT "hailo-all" (that is for the
                                              # older AI HAT+/AI Kit; the two cannot be installed together)
sudo apt install -y rpicam-apps python3-picamera2 python3-venv libegl1 libgles2 libgl1
python3 -m venv --system-site-packages .venv  # so Python can see picamera2 and hailo_platform (from apt)
source .venv/bin/activate
pip install -e ".[mediapipe,ml,dev]"
handpredict download-models                   # MediaPipe hand_landmarker.task + Hailo hand_landmark_lite.hef
```

After rebooting, check:

```bash
source .venv/bin/activate
hailortcli fw-control identify                 # names the Hailo-10H device
rpicam-hello --list-cameras                    # lists the camera (imx708 = Camera Module 3)
handpredict check-hardware --hef models/detectors/hand_landmark_lite_hailo10h.hef
```

The Hailo model file must match the installed HailoRT version. The download URL in
`configs/default.yaml` (`detector.hailo.hef_url`) points at Hailo Model Zoo **v5.4.0** for the Hailo-10H; if
`check-hardware` reports a HEF version problem, change the version in that URL to your HailoRT version
(`hailortcli --version`) and run `handpredict download-models hailo --force`.

Training the networks is faster on a laptop (see section 8). You can do everything on the Pi, it just takes longer.

---

## 5. Setting up the measurement area

```
                        camera on tripod, lens 1.00 m from the movement plane,
                        pointing straight at it (not at an angle)
                                   [CAM]
                                     |
                                     | 1 m (measure with the tape)
                                     v
   +--------------------------------------------------------------------+
   |  plain matte background                                             |
   |                                      (T1)  target 1                  |
   |                                                                      |
   |     ( S )  start region   ------------>  (T2)  target 2              |
   |                                                                      |
   |     |<-------- 50 cm tape for calibration -------->|  (T3) target 3  |
   +--------------------------------------------------------------------+
```

- **Distance and angle** are controlled variables: measure them once, write them down
  (`handpredict record --camera-distance-cm 100 --camera-notes "..."` stores them in every session file) and
  never move the camera during the study. If it moves, calibrate again and start a new dataset.
- **Targets** ~30-40 cm from the start region, in clearly different directions. Mark them with tape.
- **Lighting**: bright and even. The default exposure is a short **4 ms** to avoid motion blur, which needs
  a lot of light. Avoid fluorescent tubes and dimmed LEDs: they flicker. Same lighting for every session.
- **Background**: plain, matte, contrasting with skin; no other hands or faces in view.
- **Participant**: same hand for every trial, sleeves rolled up, no watch or rings. The tracked point is the
  middle-finger knuckle (MediaPipe landmark 9), which moves less than a fingertip when fingers curl.
- **Camera settings** (all in `configs/default.yaml`, all written to `session.json`): 1280x720 at **60 fps**
  (30 fps would put only 1.5 frames inside a 50 ms horizon), fixed exposure and gain, white balance measured
  once and then locked, manual focus at 1 m (lens position 1.0 dioptre). Camera Module 3 only reaches 60 fps
  in its 1536x864 sensor mode, which the config selects.

---

## 6. Calibration

Lay the tape flat in the movement plane. Then:

```bash
# With a screen: click two tape marks (twice: one horizontal, one vertical segment), the start region and the targets
handpredict calibrate --distance-cm 50 --distance-cm 50 --capture-targets --capture-start

# Over SSH without a screen: the frame is saved as an image; read the pixel coordinates on another computer
handpredict calibrate --points 312,640,937,642 --distance-cm 50 --points 1150,90,1152,690 --distance-cm 48 \
    --start 330,360,60 --capture-targets
```

- Two tape segments are averaged; if they disagree by more than 5 % you get a warning (camera not square to the plane).
- **Use `--capture-targets`**: the participant holds the hand at each target for a second and the tool records
  where the *knuckle* is. The physics model needs the knuckle's final position, not the paper marker's.
- The file is saved under `data/calibration/`. Use the same calibration for all sessions recorded with this setup.

---

## 7. Recording a data-collection session

```bash
handpredict record --participant P01 --calibration data/calibration/calibration_<time>.json \
    --camera-distance-cm 100 [--detector hailo] [--preview]
```

The recorder walks through a **shuffled plan** (every speed x target combination, `recording.protocol.repetitions`
times; a different order for every session):

```
Trial 7/36  speed=fast  target=T2   hand detected at (331, 362) px, in start region: yes
[Enter] record  [l] change labels  [r] reject previous  [s] skip  [q] quit >
3... 2... 1...          (recording starts: participant holds still)
GO!                     (participant moves to the target, stops, and keeps still)
Trial 7 saved (trial_007.csv): 181 frames, 1.1 % bad, 0 dropped, 60.0 fps -> OK
```

### Protocol checklist

- [ ] **Ethics first.** Studies with human participants at science fairs usually need approval
      (IRB / SRC) **before** any data is collected, and informed consent (parental consent for minors).
      Check your fair's rules.
- [ ] **Pseudonymous IDs only** (P01, P02, ...). Keep the list linking IDs to names on paper, not in this folder.
      By default **no images are stored**, only landmark coordinates. `--save-video` stores video (needed only
      for `replay`); use it only with consent.
- [ ] Same lighting, background, camera position, resolution, frame rate and detector for every session.
      `build-dataset` warns if sessions differ.
- [ ] Run `handpredict check-hardware` at the start of every session.
- [ ] Explain the task: start in the start region, hold still until **GO**, move to the announced target at the
      announced speed, stop on it, keep still until the recording ends (3 s per trial by default).
- [ ] Speeds (practise a few times first): **slow** ~1.5 s, **medium** ~0.8 s, **fast** = as fast as comfortable (~0.5 s).
- [ ] **How many trials?** The proposal's minimum is 30, but 30 % of trials go to the test set, and with
      9 test trials a difference can only just reach significance after correcting for 7 horizons (with 8 it
      cannot at all; the `summary.md` tells you when this happens). **Aim for at least 45 trials per participant**
      (3 speeds x 3 targets x 5 repetitions), and if possible several sessions or participants.
- [ ] A short break every 10-15 trials (fatigue changes movements).
- [ ] If something went wrong (hand left the frame, false start, wrong target), press **[r]** to flag the previous
      trial as rejected and redo it. Raw files are never deleted or edited; the flag is stored separately.
- [ ] Watch the quality line after each trial: more than 15 % bad frames -> redo.
- [ ] **Back up `data/raw/` to the USB stick after every session.** Raw files are read-only on purpose.

### What gets stored

Each session folder `data/raw/S<date>_<time>_<participant>/` contains:

- `session.json`: camera settings actually used (read back from the camera), camera pose, detector backend and
  the hardware it ran on (including any fallback), model checksums, calibration, the full config, software
  versions, git commit, computer model, the trial plan.
- `trial_NNN.csv`, one row per frame: frame index, timestamp (camera sensor clock), x/y in pixels and cm,
  detection confidence, `detected` and `interpolated` flags, frames dropped before it, detector latency.
  Gaps of up to 3 missing frames are interpolated and flagged; longer gaps stay empty.
- `trial_NNN.json`: labels (speed, target, notes), quality numbers, checksum of the CSV.
- `trials_index.csv`, `trial_flags.csv` (rejections), `session_end.json`.

---

## 8. From recordings to results

```bash
handpredict build-dataset                       # all sessions in data/raw -> data/processed/main
#   -> look at data/processed/main/segmentation_plots/*.png and the warnings BEFORE anything else
handpredict train --dataset data/processed/main # development trials only -> models/main
handpredict evaluate --dataset data/processed/main  # the test set -> results/main/summary.md
```

1. **`build-dataset`** checks that sessions match (resolution, fps, camera pose, landmark, detector), drops
   rejected or low-quality trials (with the reason), runs the causal filter, finds movement onset/offset and
   splits the trials into **development** (training) and **test** trials. It also reports the measured jitter of
   a still hand; if it differs a lot from `kinematics.kalman_ca.measurement_std_cm`, adjust that setting and
   rebuild. **Check the segmentation plots now**, before any prediction error has been looked at: the green line
   (onset) and purple line (offset) should bracket the movement. If they don't, adjust the thresholds in
   `kinematics.segmentation` and rebuild.
2. **`train`** uses only development trials: cross-validation chooses the network size, then 5 copies with
   different random seeds are trained and averaged. A full default run took about 10-25 minutes on a 4-core
   laptop for ~30 trials (more on the Pi). Copy `data/processed/` to the laptop, train, copy `models/` back.
3. **`evaluate`** scores every model on exactly the same test frames and writes `results/main/`:
   `summary.md`, `predictions.csv`, `trial_metrics.csv`, `model_metrics.csv`, `paired_tests.csv`,
   `crossover.json` and `plots/`.

> **The test set is looked at once.** Decide everything (thresholds, model settings, which models are the
> headline) *before* running `evaluate` on the test set. Every run is counted in `test_set_access_log.csv` and
> the count is printed in the summary. To explore while developing, use `evaluate --split dev` (clearly
> labelled as not a result: the ML models were trained on those trials).

If you record more sessions later, run `build-dataset` again: trials that already have a split keep it
(the test set never reshuffles), new trials are assigned in the same proportions. Then retrain.

---

## 9. How to read the results

**The numbers.** Error = straight-line (Euclidean) distance in cm between the predicted position and the measured
position at t + horizon. For each test trial the frame errors are averaged first (frames of one trial are not
independent), then statistics are computed across trials:

- **mean [95 % CI]**: average error; the confidence interval comes from the *bootstrap* (resampling whole trials
  thousands of times). Narrow interval = precise estimate.
- **ML - physics**: negative means the ML model was better.
- **p (Holm)**: Wilcoxon signed-rank test on the per-trial differences (does ML beat physics consistently across
  trials?), corrected with Holm's method because 7 horizons are tested. Below 0.05 = "significant".
- **effect r**: -1 = ML better on every trial, +1 = physics better on every trial, 0 = no tendency.
- **crossover**: the horizon where the difference changes sign (physics better before, ML better after), with a
  bootstrap interval. The hypothesis predicts such a crossover.
- **trend**: does the ML advantage grow with the horizon? (negative slope = yes).

**The analysis windows.** *movement_strict* (primary, as in the proposal): predictions made during the movement
whose target time is still inside the movement. Long horizons then only count early parts of slower movements,
so the horizon effect is mixed with a speed effect; *movement_any_target* (target time may be after the hand has
stopped) does not have this problem. *whole_trial* includes the resting periods.

**The verdict** in `summary.md` follows fixed rules written before any data existed
(`handpredict/evaluation/summary.py`): **supported** only if physics is better at short horizons, ML at longer
ones, and ML's advantage is significant after the crossover. Otherwise: **partly supported** (ML better
everywhere), **contradicted** (physics significantly better at long horizons), **suggestive** or **unresolved**.

**The plots** (`results/main/plots/`, each with a CSV of the plotted numbers):

| Plot | What to look for |
|---|---|
| `error_vs_horizon_*.png` | Error grows with the horizon for every model. Where do the headline curves (thick) cross? `_all_models_` uses a log scale. |
| `difference_vs_horizon_*.png` | The hypothesis in one picture: does the line go from above 0 (physics better) to below 0 (ML better)? Stars = significant. |
| `paired_differences_*.png` | One dot per test trial: is ML better on most trials, or on a few by a lot? |
| `example_trajectories.png` | The measured path (grey) with predicted positions at 50/100/200 ms. Predictions should lie on the path, ahead of the hand. |
| `error_vs_speed.png` | Fast movements are harder; which model copes better? |
| `learning_curves_*.png` | Validation loss should flatten, not rise (rising = overfitting). |
| `overfitting_diagnostic.png` | Training-trial error far below test error = the network memorised the training data. |
| `breakdown_by_*.png` | Differences between speed conditions / targets (descriptive only: few trials per group). |

---

## 10. Edge AI on the AI HAT+ 2

**What runs where** is recorded in every session (`session.json -> detector`) and every results folder:

| Part | Default | With the AI HAT+ 2 |
|---|---|---|
| Hand-landmark detection | MediaPipe on the CPU | `--detector hailo`: Hailo Model Zoo `hand_landmark_lite` on the Hailo-10H |
| ML predictor (GRU/MLP) | ONNX Runtime on the CPU | stretch goal: MLP compiled for the Hailo-10H (`export-hailo`) |
| Physics predictor | CPU | CPU |

**Hailo detector.** The Hailo model zoo provides MediaPipe's landmark network compiled for the Hailo-10H, but no
hand *finder*. So the detector looks for the hand in the **calibrated start region** (trying several rotations and
sizes), then follows it from frame to frame using the previous landmarks (MediaPipe's own method). If the Hailo
device is missing, it falls back to MediaPipe and logs `Hailo device not found; falling back to CPU detector`;
`session.json` records `fallback_from: hailo`. A controlled variable: use one detector for the whole study.
To compare detectors, record with `--save-video` and run `handpredict replay --session ... --detector hailo`.

**Benchmark.** Run on the Pi with the real camera:

```bash
handpredict benchmark --models-dir models/main --calibration data/calibration/<file>.json --detector hailo
handpredict benchmark --models-dir models/main --calibration data/calibration/<file>.json --detector mediapipe
```

It reports each predictor's latency and the live pipeline stage by stage (camera wait, detection, filter,
physics, ML), frames per second, dropped frames, CPU use and temperature, and the **capture-to-prediction
latency**. A prediction for 50 ms ahead is only useful if it is ready in less than 50 ms: the report says how
much time is left, or that it arrives too late. That is a strong point for the write-up.

**ML predictor on the Hailo-10H (stretch goal).** The compiler (Hailo Dataflow Compiler, free download from the
Hailo Developer Zone) runs on an **x86-64 Linux computer**, not on the Pi, and supports recurrent layers only in a
limited way, which is why the dense **MLP** exists:

```bash
handpredict export-hailo --models-dir models/main --step onnx       # any computer: mlp_flat.onnx + calibration data
handpredict export-hailo --models-dir models/main --step compile    # x86 Linux with the DFC: mlp.hef
handpredict benchmark --models-dir models/main --ml-hef models/main/mlp/hailo/mlp.hef --predictors-only   # on the Pi
```

The benchmark also reports how much the accelerator's 8-bit outputs differ from the original model. Only claim
the predictor ran on the accelerator if you did this step.

---

## 11. Live demo

```bash
handpredict demo --calibration data/calibration/<file>.json --models-dir models/main --target T2 [--detector hailo]
```

Green dot = the knuckle now; blue ring = physics prediction; orange cross = ML prediction, both *h* ms ahead.
The top line shows each model's rolling error over the last 2 s (each prediction is scored when its moment
arrives). Keys 1-7 change the horizon, q quits. Without a screen it prints the same line every second;
`--output demo.mp4` also writes the overlay video (it shows the participant: consent!).

---

## 12. Design decisions and assumptions

### Fixed before any data was collected (do not change them after looking at test results)

- **Headline physics model: `minimum_jerk_euler`.** Flash & Hogan's minimum-jerk path from the resting start
  point to the calibrated target, with onset time t0 and duration T fitted online to the samples seen so far
  (grid search + bounded least squares), rolled forward with the proposal's Newton-Euler steps
  `v[n+1] = v[n] + a[n] dt, x[n+1] = x[n] + v[n] dt` using the minimum-jerk acceleration. It falls back to
  constant velocity (and logs why) until the fit is trustworthy, and holds at the target once the movement is over.
  A test proves the Euler-rolled and closed-form versions agree to within 0.1 % of the reach distance.
- **Headline ML model: `gru`**, the average of 5 GRU networks trained with different random seeds.
- Other rows are **ablations and baselines**, not alternatives to pick from after the fact: zero motion (the floor
  every model must beat), constant velocity, Newton-Euler with constant acceleration, closed-form minimum jerk,
  minimum jerk with a free (fitted) end point, MLP, Gaussian process, and **`minimum_jerk_replan`**
  (Hoff & Arbib 1993: re-plan the minimum-jerk path from the hand's *measured* position and velocity). On
  synthetic data the re-planning variant beats the headline at every horizon, because real paths curve slightly
  and the straight-line model can be off sideways even "now". If you prefer it as the headline, decide that
  **before collecting data** (change `physics.headline`) and say so in the write-up.

### Fairness and leakage

- **Causal everything.** Predictions at time t use only samples at or before t. The filters run forward in time
  only; predictors receive a `History` that physically contains only past samples. Tests change all future
  samples and check that no earlier output changes, for every model.
- **Same information for both sides.** The minimum-jerk model is given the target, so the ML model gets the
  target position (relative to the hand) as an input too; both see the same history, frames and ground truth.
  The physics model uses all samples since the movement started; the ML model a fixed window (`ml.history_ms`,
  200 ms). Ablation flags remove the target or the absolute position from the ML inputs.
- **Split by whole trials** (stratified by speed x target; by participant if there are at least 3). Test trials
  are never loaded during training (a test checks this). Normalisation statistics come from training data only.
  Augmentation (mild time-stretching, small jitter) only touches training data.
- **Ground truth** = the measured knuckle position at t + h, linearly interpolated between real detections (never
  filtered or interpolated values). Detector jitter (~0.1-0.2 cm) therefore sets an error floor that is the same
  for every model.
- **Trials, not frames,** are the statistical unit; bootstrap CIs resample whole trials.

### Assumptions (each one is a config value)

| Assumption | Config |
|---|---|
| Raspberry Pi 5, **Raspberry Pi OS Trixie** (required by the AI HAT+ 2 software; Python 3.13), `picamera2` for the Pi camera, OpenCV for webcams/videos | `camera.source` |
| 2-D tracking in the image plane, converted to cm with one scale factor; MediaPipe's relative z is logged but not used | `landmark`, calibration |
| Tracked landmark: middle-finger MCP knuckle (MediaPipe 9), fixed for the whole study | `landmark.index` |
| 1280x720 at 60 fps (1536x864 sensor mode), fixed exposure 4 ms and gain, locked white balance, manual focus at 1 m | `camera.*` |
| One hand, one participant at a time, pseudonymous IDs, no images stored by default | `recording.save_video` |
| Filter and segmentation settings chosen on synthetic reaches before real data existed (a still hand never triggers a false onset) | `kinematics.*` |
| Horizons 50, 100, 150, 200, 300, 400, 500 ms | `horizons_ms` |

Changes from the original brief, and why:

- **OS Trixie, `hailo-h10-all`** instead of Bookworm: required by the current official AI HAT+ 2 documentation.
- **MediaPipe Tasks API** (`hand_landmarker.task`): the old `mp.solutions.hands` API no longer exists in MediaPipe 1.x.
- **Hailo detector = landmark network + crop tracking**: the Hailo Model Zoo has the landmark network for the
  Hailo-10H but no palm detector.
- **Extra analysis window** (`movement_any_target`) because the primary window mixes horizon with speed.
- **Onset threshold 10 cm/s** (not lower): detector jitter makes a still hand look like it moves at a few cm/s.

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `picamera2 is not available` | `sudo apt install python3-picamera2` and recreate the venv with `--system-site-packages`. |
| `No Raspberry Pi camera detected` | Check the ribbon (contacts facing the right way, latch closed); `rpicam-hello --list-cameras`. |
| Camera runs below 60 fps | Keep `camera.sensor_output_size: [1536, 864]`; exposure must be shorter than 16.6 ms; enough light. |
| Images too dark / blurry | More light (exposure is short on purpose); check focus (`camera.focus.lens_position_dioptres` = 1 / distance in m). |
| `Hailo device not found; falling back to CPU detector` | `hailortcli fw-control identify`; `dmesg \| grep -i hailo`; reseat the HAT; reinstall `hailo-h10-all`; reboot. |
| HEF version error | Download the HEF matching `hailortcli --version` (section 4). |
| Hailo detector misses the hand at the start | The hand must start inside the start region; enlarge `detector.hailo.reacquire_size_px`, or use `detector.hailo.reacquire: mediapipe`. |
| Wrong Hailo outputs (landmarks nonsense) | `handpredict check-hardware --hef <file>` lists the outputs; set `detector.hailo.output_map: {landmarks: <name>, presence: <name>}`. |
| `libEGL.so.1: cannot open shared object file` | `sudo apt install libegl1 libgles2` (MediaPipe needs it). |
| `libGL.so.1` missing | `sudo apt install libgl1`. |
| `No display available` during calibration | Use `--points` / `--start` / `--target` (section 6), or connect a screen / use VNC. |
| MediaPipe too slow on the Pi (dropped frames) | Expected at 60 fps on the CPU: the recorder buffers frames so none are lost, but use the Hailo detector for live use. |
| Too many false movement onsets / onset detected late | Look at the segmentation plots; adjust `kinematics.segmentation.*` and `kinematics.kalman_ca.measurement_std_cm`, rebuild. Never tune them on test results. |
| `checksum mismatch` | A raw file was edited. Restore it from the backup; raw data must not be changed. |
| `Refusing to mix synthetic and real sessions` | Keep synthetic sessions out of `data/raw` (the self-test uses its own folder). |
| `the config's horizons differ ...` | You changed `horizons_ms` after training: retrain. |
| A value like `2e6` in the config is read as text | It isn't here: the loader accepts YAML 1.2 numbers. Check spelling of the key instead (unknown keys are rejected). |

---

## 14. For developers: tests, reproducibility, code map

- `pytest` runs 100+ tests on a laptop without hardware (about 2 minutes). Tests that need the downloaded MediaPipe
  model or `ai-edge-litert` skip themselves if those are missing (`handpredict download-models mediapipe test-assets`,
  `pip install ai-edge-litert`). `pytest -m "not slow"` skips the end-to-end pipeline test.
- Among them: Euler vs closed-form minimum jerk, noiseless recovery, constant velocity exact on constant-velocity
  motion, causal filters (future perturbation), online == offline filtering and prediction, ground-truth
  interpolation, split overlap and stickiness, training never touching test trials, window shapes, ONNX ==
  PyTorch, hand-computed metrics/Holm/Wilcoxon/bootstrap, verdict rules, an end-to-end leak test for every model,
  real MediaPipe inference on a photo and on a moving-hand video, and the Hailo detector's cropping, decoding and
  tracking logic (run on the same landmark network on the CPU).
- **Reproducibility:** one seed (`study.seed`) drives every random choice; every session, model folder and results
  folder stores the full config, its hash, the git commit, package versions and the computer description.
  `requirements-lock.txt` pins the exact laptop environment used during development.
- Lines that could not be tested without the AI HAT+ 2 or the Dataflow Compiler are marked `VERIFY ON HARDWARE`
  / `VERIFY ON DFC MACHINE`.

---

## 15. References

- Flash, T. & Hogan, N. (1985). The coordination of arm movements: an experimentally confirmed mathematical model. *Journal of Neuroscience* 5(7), 1688-1703.
- Hoff, B. & Arbib, M. A. (1993). Models of trajectory formation and temporal interaction of reach and grasp. *Journal of Motor Behavior* 25(3), 175-192.
- Gamage, N. M., Ishtaweera, D., Weigel, M. & Withana, A. (2021). So Predictable! Continuous 3D Hand Trajectory Prediction in Virtual Reality. *UIST '21*.
- Zhang, F. et al. (2020). MediaPipe Hands: On-device real-time hand tracking. arXiv:2006.10214.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin* 1(6), 80-83.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 65-70.
- Kerby, D. S. (2014). The simple difference formula: an approach to teaching nonparametric correlation. *Comprehensive Psychology* 3.
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Raspberry Pi Ltd. *AI HAT+ and AI HAT+ 2 documentation* (raspberrypi.com/documentation/computers/ai.html); Hailo, *hailo-apps*, *Hailo Model Zoo* and *HailoRT* (github.com/hailo-ai).
