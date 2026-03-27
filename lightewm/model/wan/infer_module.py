from lightewm.runner.pipeline_factory import build_wan_i2v_infer_pipeline


class WanI2VInferModel:
    def __init__(self, config):
        model_params = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        self.pipe = build_wan_i2v_infer_pipeline(model_params)

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)
