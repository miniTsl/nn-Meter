# initialize builder config with workspace
from nn_meter.builder import builder_config
workspace = "/mnt/sda/nn-Meter/workspaces/pytorch_gpu"
builder_config.init(workspace)

# # build kernel predictor
# from nn_meter.builder.backends import connect_backend
# backend = connect_backend(backend_name='pytorch_gpu')
# from nn_meter.builder.kernel_predictor_builder import generate_config_sample, build_predictor_by_data
# from nn_meter.builder import profile_models
# import os
# kernel_type = "conv-bn-relu"
# sample_num = 2000
# mark = "test"
#
# # sample configs for kernel and generate models
# models = generate_config_sample(kernel_type, sample_num, mark=mark,
#                                 sampling_mode="prior")
#
# # # convert the model to the needed format by backend, in order to increase efficiency when profiling on device.
# # models = convert_models(backend, f"{workspace}/predictor_build/results/{kernel_type}_{mark}.json")
#
# # run models with given backend and return latency of testcase models
# profiled_results = profile_models(backend, models, mode='predbuild', have_converted=True,
#                                   save_name=f"profiled_{kernel_type}.json")
#
# error_threshold = 0.1
#
# # extract training feature and target from profiled results
# cfgs_path = os.path.join(workspace, "predictor_build", "results", f"{kernel_type}_{mark}.json")
# lats_path = os.path.join(workspace, "predictor_build", "results", f"profiled_{kernel_type}.json")
# kernel_data = (cfgs_path, lats_path)
#
# # build latency predictor
# predictor, acc10, error_configs = build_predictor_by_data(
#     kernel_type, kernel_data, backend, error_threshold=error_threshold, mark=mark,
#     save_path=os.path.join(workspace, "predictor_build", "results")
# )
# print(f'Iteration 0: acc10 {acc10}, error_configs number: {len(error_configs)}')

# # build latency predictor for all kernels in config
# from nn_meter.builder import build_latency_predictor
# build_latency_predictor(backend="pytorch_gpu")

from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data
import json
import os
workspace_path = builder_config.get('WORKSPACE', 'predbuild')
kernel_type = "add"
mark = ""
backend = "pytorch_gpu"
json_path = os.path.join(workspace_path, "results", f"profiled_{kernel_type}.json")
print(json_path)


# use current sampled data to build regression model, and locate data with large errors in testset
# add save_path for torch implementation to save data.csv and predictor.pkl
# the backend parameter is used to determine the implementation of predictor
predictor, acc10, error_configs = build_predictor_by_data(kernel_type, json_path, backend,
                                                          error_threshold=0.1, mark=f'customized{mark}',
                                                          save_path=os.path.join(workspace_path, "results"),
                                                          predict_label="latency")
