#!/usr/bin/env bash
# Set up a Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H) + Camera Module 3 for this project.
#
# Run from the repository folder on Raspberry Pi OS (64-bit, Trixie):
#     bash scripts/setup_pi.sh
# The package names follow the official Raspberry Pi AI documentation
# (https://www.raspberrypi.com/documentation/computers/ai.html, checked Oct 2026).
# Reboot when the script asks you to, then run it again: it skips what is done.
set -euo pipefail

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This script is for the Raspberry Pi (64-bit). On a laptop just run: pip install -e '.[mediapipe,ml,dev]'"
    exit 1
fi
. /etc/os-release
if [[ "${VERSION_CODENAME:-}" != "trixie" ]]; then
    echo "WARNING: the AI HAT+ 2 software needs Raspberry Pi OS Trixie; this is '${VERSION_CODENAME:-unknown}'."
fi

say "1/5 Updating Raspberry Pi OS and firmware"
sudo apt update
sudo apt full-upgrade -y
sudo rpi-eeprom-update -a || true

say "2/5 Installing the Hailo-10H software (AI HAT+ 2)"
if dpkg -s hailo-all >/dev/null 2>&1; then
    echo "ERROR: 'hailo-all' (for the older AI HAT+/AI Kit) is installed. It cannot coexist with 'hailo-h10-all'."
    echo "Remove it first:  sudo apt remove hailo-all  (then run this script again)"
    exit 1
fi
sudo apt install -y dkms
sudo apt install -y hailo-h10-all

say "3/5 Installing the camera stack and system libraries"
# libegl1/libgles2: needed by MediaPipe; libgl1: needed by OpenCV's window functions.
sudo apt install -y rpicam-apps python3-picamera2 python3-venv python3-dev libegl1 libgles2 libgl1

say "4/5 Creating the Python environment (.venv)"
# --system-site-packages lets Python see picamera2 and hailo_platform, which come from apt.
if [[ ! -d .venv ]]; then
    python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[mediapipe,ml,dev]"

say "5/5 Downloading the hand-landmark models"
handpredict download-models

cat <<'EOF'

Done. If this was the first run, REBOOT now (sudo reboot), then check everything:

    source .venv/bin/activate
    hailortcli fw-control identify          # should name the Hailo-10H
    rpicam-hello --list-cameras             # should list the camera (imx708 = Camera Module 3)
    handpredict check-hardware --hef models/detectors/hand_landmark_lite_hailo10h.hef
    handpredict selftest                    # whole pipeline on synthetic data, a few minutes
EOF
