from lightewm.runner.runner_util.instantiation import instantiate_component_from_section


class BaseRunnerPipeline:
    def __init__(self, config):
        self.config = config

    def run(self):
        runner, _ = instantiate_component_from_section(self.config.runner, self.config, section_name="runner")
        if not hasattr(runner, "run"):
            raise TypeError(f"Runner '{self.config.runner.class_path}' has no run() method")
        return runner.run()
