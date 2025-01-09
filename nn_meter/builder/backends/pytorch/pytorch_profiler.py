# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, schedule
from nn_meter.builder.backends import BaseProfiler
import time

class PytorchProfiler(BaseProfiler):

    use_cuda = False

    def __init__(self, weight_hub_path=None, num_runs=50, warm_ups=20):
        """__init__ function for PytorchProfiler

        Args:
            weight_hub_path (str): path to the overall weight file
            num_runs (int, optional): _description_. Defaults to 50.
            warm_ups (int, optional): _description_. Defaults to 10.
        """
        self._num_runs = num_runs
        self._warm_ups = warm_ups

    def profile(self, model_path, input_shape, data_type='float32', **kwargs):
        """profile function for PytorchProfiler

        Args:
            model_path (str): path to the script model file
            input_shape: list of input_shape of the input data
            data_type (str, optional): data type of input data. Defaults to 'float32'.

        Returns:
            output: the textual output of the torch.profile tool
        """        
        model = torch.jit.load(model_path)
        model.eval()

        batch_size = 1
        if len(input_shape) == 1:
            input_data = [torch.randn(size=[batch_size] + input_shape[0])]
        else:
            input_data = [torch.randn(size=[batch_size] + shape) for shape in input_shape]

        # decide whether to move input_data and model onto cuda, default GPU number is 0
        if self.use_cuda:
            # check if cuda is available
            try :
                assert torch.cuda.is_available()
            except AssertionError:
                raise Exception("CUDA is not available.")
            cuda_id = 0
            # if len(input_data) == 1:
            #     input_data = input_data.cuda(cuda_id)
            # else:
            input_data = [data.cuda(cuda_id) for data in input_data]

            model = model.cuda(cuda_id)

            # # using torch.profile to profile the model
            # activites = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            # # At the end of each cycle profiler calls the specified on_trace_ready function and passes itself as an argument.
            # def trace_handler(p):
            #     output = p.key_averages().table(sort_by="cuda_time_total")
            #     # save the output to a local txt file
            #     output_file = 'output.txt'
            #     with open(output_file, 'w') as f:
            #         f.write(output)
        else:
            pass

            # # using torch.profile to profile the model
            # activites = [ProfilerActivity.CPU]
            # def trace_handler(p):
            #     output = p.key_averages().table(sort_by="cpu_time_total")
            #     # save the output to a local txt file
            #     output_file = 'output.txt'
            #     with open(output_file, 'w') as f:
            #         f.write(output)

        # # using torch.profile to profile the model
        # my_schedule = schedule(
        #     wait=0,
        #     warmup=self._warm_ups,
        #     active=self._num_runs,
        #     repeat=1)
        # with profile(activities=activites, schedule=my_schedule, on_trace_ready=trace_handler) as p:
        #     for idx in range(self._warm_ups + self._num_runs):
        #         model(*input_data)
        #         p.step()
        # # read output from the local txt file and then remove it
        # output_file = 'output.txt'
        # with open(output_file, 'r') as f:
        #     output = f.read()
        # os.remove(output_file)

        for _ in range(self._warm_ups):
            model(*input_data)

        run_time = []
        for index in range(self._num_runs):
            start_time = time.perf_counter_ns()
            model(*input_data)
            end_time = time.perf_counter_ns()
            run_time.append((end_time - start_time)/1e6) # convert to ms

        output = {}
        output['latency_avg'] = sum(run_time) / len(run_time)
        output['latency_max'] = max(run_time)
        output['latency_min'] = min(run_time)
        # compute std using numpy
        import numpy as np
        output['latency_std'] = np.std(run_time)
        return output

if __name__ == '__main__':
    
    # use pre-trained resnet18 model as an example
    model = models.resnet18()
    weight_path = '/mnt/sda/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    
    profiler = PytorchProfiler(model, weight_path, data_type='float32')
    input_shape = [[1, 3, 224, 224]]
    output = profiler.profile(input_shape)
    print(output)