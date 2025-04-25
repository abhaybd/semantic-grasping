#!/bin/bash

set -euo pipefail

mkdir -p deps
cd deps
git clone -b graspmolmo_eval --single-branch https://github.com/allenai/robo_mm_olmo.git
cd robo_mm_olmo
pip install -e .[all]
