#!/usr/bin/env bash
set -euo pipefail

ROOT="${LIGHTEWM_EVAL_ROOT:-/mnt/data/zhangyu/robot-eval}"
LIGHTEWM_ROOT="${LIGHTEWM_ROOT:-${ROOT}/repos/LightEWM}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${ROOT}/repos/RoboTwin}"
ROBOLAB_ROOT="${ROBOLAB_ROOT:-${ROOT}/repos/RoboLab}"
STARWAM_ROOT="${STARWAM_ROOT:-${ROOT}/repos/StarWAM}"

apply_once() {
  local repo="$1"
  local patch="$2"
  if git -C "${repo}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
    return
  fi
  git -C "${repo}" apply --check "${patch}"
  git -C "${repo}" apply "${patch}"
}

apply_once \
  "${ROBOTWIN_ROOT}" \
  "${LIGHTEWM_ROOT}/scripts/benchmark_overlays/robotwin_eval_test_num.patch"
apply_once \
  "${ROBOLAB_ROOT}" \
  "${LIGHTEWM_ROOT}/scripts/benchmark_overlays/robolab_deterministic_hold.patch"
apply_once \
  "${ROBOLAB_ROOT}" \
  "${LIGHTEWM_ROOT}/scripts/benchmark_overlays/robolab_task_scoped_registration.patch"

ln -sfn \
  "${LIGHTEWM_ROOT}/lightewm/integrations/robotwin/control_policy" \
  "${ROBOTWIN_ROOT}/policy/lightewm_control"
ln -sfn \
  "${STARWAM_ROOT}/examples/robotwin" \
  "${ROBOTWIN_ROOT}/policy/starwam_client"

cp \
  "${LIGHTEWM_ROOT}/examples/closed_loop/overrides/robotwin_demo_clean_gate.yml" \
  "${ROBOTWIN_ROOT}/task_config/lightewm_demo_clean_gate.yml"

echo "Closed-loop benchmark overlays are ready."
