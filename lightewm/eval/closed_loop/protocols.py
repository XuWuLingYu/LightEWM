from __future__ import annotations

from dataclasses import dataclass, replace

from .harness import CommandStep


ROBOLAB_FATAL_LOG_PATTERNS = (
    "Traceback (most recent call last):",
    "Terminated with error:",
)
ROBOLAB_RUNTIME_ENV = {"OMNI_KIT_ACCEPT_EULA": "YES"}


@dataclass(frozen=True)
class RoboTwinProtocol:
    root: str
    python: str

    def render(self) -> CommandStep:
        return CommandStep(
            name="robotwin_render",
            argv=(self.python, "script/test_render.py"),
            cwd=self.root,
            required_log_patterns=("Render Well",),
            forbidden_log_patterns=("Render Error",),
        )

    def collect(self, task: str, setting: str) -> CommandStep:
        return CommandStep(
            name=f"robotwin_ground_truth_{task}_{setting}",
            argv=(self.python, "script/collect_data.py", task, setting),
            cwd=self.root,
            required_log_patterns=("Render Well", "success!"),
            forbidden_log_patterns=("Render Error", "Collect Error"),
        )

    def evaluate(
        self,
        *,
        config: str,
        policy_name: str,
        task: str,
        setting: str,
        checkpoint_tag: str,
        seed: int,
        episodes: int = 100,
        overrides: tuple[str, ...] = (),
        name: str = "robotwin_reference",
    ) -> CommandStep:
        pairs = (
            "policy_name",
            policy_name,
            "task_name",
            task,
            "task_config",
            setting,
            "instruction_type",
            "unseen",
            "ckpt_setting",
            checkpoint_tag,
            "seed",
            str(seed),
            "test_num",
            str(episodes),
            *overrides,
        )
        return CommandStep(
            name=f"{name}_{task}_{setting}",
            argv=(
                self.python,
                "script/eval_policy.py",
                "--config",
                config,
                "--overrides",
                *pairs,
            ),
            cwd=self.root,
        )

    def negative_control(
        self,
        *,
        config: str,
        task: str,
        setting: str,
        seed: int,
        mode: str,
        probe_path: str,
    ) -> CommandStep:
        if mode not in {"hold", "zero", "wrong_gripper", "reverse"}:
            raise ValueError(f"unsupported RoboTwin negative control: {mode}")
        step = self.evaluate(
            config=config,
            policy_name="lightewm_control",
            task=task,
            setting=setting,
            checkpoint_tag=f"negative-{mode}",
            seed=seed,
            episodes=1,
            overrides=(
                "control_mode",
                mode,
                "control_probe_path",
                probe_path,
            ),
            name=f"robotwin_negative_{mode}",
        )
        return replace(
            step,
            expected_output=(probe_path,),
            required_log_patterns=("Fail!", "Success rate:"),
            forbidden_log_patterns=("Success!",),
        )


@dataclass(frozen=True)
class RoboLabProtocol:
    root: str
    python: str

    def recorded_replay(
        self,
        task: str,
        *,
        recorded_data_folder: str | None = None,
        episode: int = 0,
        validate_states: bool = True,
    ) -> CommandStep:
        argv = [
            self.python,
            "examples/run_recorded.py",
            "--task",
            task,
            "--episode",
            str(episode),
            "--headless",
            "--disable-subtask",
        ]
        if validate_states:
            argv.append("--validate-states")
        if recorded_data_folder:
            argv.extend(["--recorded-data-folder", recorded_data_folder])
        return CommandStep(
            name=f"robolab_recorded_replay_{task}",
            argv=tuple(argv),
            cwd=self.root,
            env=dict(ROBOLAB_RUNTIME_ENV),
            required_log_patterns=(
                "Restored recorded env config",
                "STATE VALIDATION:",
            ),
            forbidden_log_patterns=(
                "WARNING: no recorded initial state",
                "WARNING: cannot validate states",
                "STATE VALIDATION: replay diverged",
                *ROBOLAB_FATAL_LOG_PATTERNS,
            ),
        )

    def gripper_toggle(self, task: str) -> CommandStep:
        return CommandStep(
            name=f"robolab_gripper_toggle_{task}",
            argv=(
                self.python,
                "examples/run_gripper_toggle.py",
                "--task",
                task,
                "--headless",
            ),
            cwd=self.root,
            env=dict(ROBOLAB_RUNTIME_ENV),
            expected_output=(
                f"output/run_gripper_toggle/{task}/gripper_toggle_probe.json",
            ),
            forbidden_log_patterns=ROBOLAB_FATAL_LOG_PATTERNS,
        )

    def deterministic_hold(self, task: str, *, steps: int = 50) -> CommandStep:
        return CommandStep(
            name=f"robolab_deterministic_hold_{task}",
            argv=(
                self.python,
                "examples/run_empty.py",
                "--task",
                task,
                "--num-steps",
                str(steps),
                "--action-mode",
                "hold",
                "--headless",
            ),
            cwd=self.root,
            env=dict(ROBOLAB_RUNTIME_ENV),
            expected_output=(
                f"output/run_empty_env/{task}/control_probe_0.json",
                "output/run_empty_env/episode_results.jsonl",
            ),
            forbidden_log_patterns=ROBOLAB_FATAL_LOG_PATTERNS,
        )

    def reference_policy(
        self,
        task: str,
        *,
        policy: str = "pi05",
        host: str = "127.0.0.1",
        port: int = 8000,
        num_runs: int = 1,
        output_folder_name: str | None = None,
    ) -> CommandStep:
        output_folder_name = (
            output_folder_name or f"lightewm_reference_{policy}_{task}"
        )
        return CommandStep(
            name=f"robolab_reference_{policy}_{task}",
            argv=(
                self.python,
                "policies/pi0_family/run.py",
                "--policy",
                policy,
                "--task",
                task,
                "--remote-host",
                host,
                "--remote-port",
                str(port),
                "--num-runs",
                str(num_runs),
                "--num_envs",
                "1",
                "--output-folder-name",
                output_folder_name,
                "--headless",
            ),
            cwd=self.root,
            env=dict(ROBOLAB_RUNTIME_ENV),
            expected_output=(
                f"output/{output_folder_name}/episode_results.jsonl",
            ),
            forbidden_log_patterns=ROBOLAB_FATAL_LOG_PATTERNS,
        )
