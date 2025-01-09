# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from nn_meter.builder.backends.pytorch.pytorch_profiler import PytorchProfiler
from nn_meter.builder.backends.pytorch.pytorch_backend import PytorchBackend
from nn_meter.builder.backends import BaseParser
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults
import torchvision.models as models

class PytorchGPULatencyParser(BaseParser):

    def parse(self, content):
        self.gpu_time = Latency(avg=content['latency_avg'], std=content['latency_std'])
        self.gpu_time_max = content['latency_max']
        self.gpu_time_min = content['latency_min']

        # using torch.profile to profile the model
        # self.layers = self._parse_layers(content)
        # # the first line of pytorch profiler
        # self.runs = self.layers[0]["# of Calls"]
        # # for profiler using cuda, the first layer of CPU is always empty
        # self.cpu_total_avg = self.layers[1]["CPU time avg"]
        # # sum all the layers' self cpu time
        # self.self_cpu_total = sum(
        #     Latency(layer['Self CPU'])
        #     for layer in self.layers
        # )
        # self.self_cpu_total.avg = self.self_cpu_total.avg/float(self.runs)
        #
        # self.cuda_total_avg = self.layers[0]["CUDA time avg"]
        # # sum all the layers' self cuda time
        # self.self_cuda_total = sum(
        #     Latency(layer['Self CUDA'])
        #     for layer in self.layers[1:]
        # )
        # self.self_cuda_total.avg = self.self_cuda_total.avg/float(self.runs)
        return self
        
    # time: ms
    def convert_to_ms(self, value):
        if value.endswith('us'):
            return float(value[:-2]) / 1000  
        elif value.endswith('ms'):
            return float(value[:-2])  
        return float(value)  

    def _parse_layers(self,content):
        # match every line, skip title and empty lines
        # notice the time can be in ms or us
        columns = ["Name", "Self CPU %", "Self CPU (ms)", "CPU total %", "CPU total (ms)", "CPU time avg (ms)", "Self CUDA (ms)", "Self CUDA %", "CUDA total (ms)", "CUDA time avg (ms)", "# of Calls"]
        
        # match every line, skip title and empty lines
        # notice the time can be in ms or us
        pattern = re.compile(r'(\S+.*?)\s+([\d.]+%)\s+([\d.]+[mu]s)\s+([\d.]+%)\s+([\d.]+[mu]s)\s+([\d.]+[mu]s)\s+([\d.]+[mu]s)\s+([\d.]+%)\s+([\d.]+[mu]s)\s+([\d.]+[mu]s)\s+(\d+)')
    
        # fetch every line
        matches = pattern.findall(content)
        
        layers = []
        for match in matches:
            row_dict = {
                "Name": match[0],
                "Self CPU %": float(match[1][:-1]),
                "Self CPU": self.convert_to_ms(match[2]),
                "CPU total %": float(match[3][:-1]),
                "CPU total": self.convert_to_ms(match[4]),
                "CPU time avg": self.convert_to_ms(match[5]),
                "Self CUDA": self.convert_to_ms(match[6]),
                "Self CUDA %": float(match[7][:-1]),
                "CUDA total": self.convert_to_ms(match[8]),
                "CUDA time avg": self.convert_to_ms(match[9]),
                "# of Calls": int(match[10])
            }
            layers.append(row_dict)
        
        return layers

    @property
    def latency(self):
        return self.gpu_time
        # return self.self_cpu_total + self.self_cuda_total
    
    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results



class PytorchGPUProfiler(PytorchProfiler):
    use_cuda = True


class PytorchGPUBackend(PytorchBackend):
    parser_class = PytorchGPULatencyParser
    profiler_class = PytorchGPUProfiler

if __name__ == '__main__':
    # use pre-trained resnet18 model as an example
    model = models.resnet18()
    weight_path = '/mnt/sda/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    
    profiler = PytorchGPUProfiler(model, weight_path)
    shapes = [[1, 3, 224, 224]]
    output = profiler.profile(shapes)
    print(output)
    
    parser = PytorchGPULatencyParser()
    parser.parse(output)
    
    print(parser.runs)
    print(parser.cpu_total_avg)
    print(parser.self_cpu_total)
    print(parser.cuda_total_avg)
    print(parser.self_cuda_total)
