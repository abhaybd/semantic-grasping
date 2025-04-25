#!/bin/bash

set -euo pipefail

mkdir -p deps
cd deps
# git clone -b graspmolmo_eval --single-branch https://github.com/allenai/robo_mm_olmo.git
ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts
git clone -b graspmolmo_eval --single-branch git@github.com:allenai/robo_mm_olmo.git
cd robo_mm_olmo
pip install -e .[all]
