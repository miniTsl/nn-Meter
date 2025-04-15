# initialize builder config with workspace
from nn_meter.builder import builder_config
workspace = "/mnt/sda/nn-Meter/workspaces/pytorch_cpu"
builder_config.init(workspace)

# generate testcases for op fusion rule testing
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases
# return contains model structure and input_shape as for torch framework
origin_testcases = generate_testcases()

# connect to backend
from nn_meter.builder.backends import connect_backend
backend = connect_backend(backend_name='pytorch_cpu')

# profile models with backend
from nn_meter.builder import profile_models
# return contains model structure, input_shape and profiling results(latency for now)
profiled_results = profile_models(backend, models=origin_testcases, mode='ruletest')

# determine fusion rules from profiling results
from nn_meter.builder.backend_meta.fusion_rule_tester import detect_fusion_rule
detected_results = detect_fusion_rule(profiled_results)
