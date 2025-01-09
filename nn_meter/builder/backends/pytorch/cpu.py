# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from pytorch_profiler import PytorchProfiler
from pytorch_backend import PytorchBackend
from nn_meter.builder.backends import BaseParser
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults
import torchvision.models as models

class PytorchCPULatencyParser(BaseParser):

    def parse(self, content):

        self.cpu_time = Latency(avg=content['latency_avg'], std=content['latency_std'])
        self.cpu_time_max = content['latency_max']
        self.cpu_time_min = content['latency_min']

        # # using torch.profile to profile the model
        # self.layers = self._parse_layers(content)
        # # the first line of pytorch profiler
        # self.runs = self.layers[0]["# of Calls"]
        # self.cpu_total_avg = self.layers[0]["CPU time avg"]
        # # sum all the layers' self cpu time
        # self.self_cpu_total = sum(
        #     Latency(layer['Self CPU'])
        #     for layer in self.layers
        # )
        # self.self_cpu_total.avg = self.self_cpu_total.avg/float(self.runs)
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
        pattern = re.compile(r'(\S+.*?)\s+([\d.]+%)\s+([\d.]+[mu]s)\s+([\d.]+%)\s+([\d.]+[mu]s)\s+([\d.]+[mu]s)\s+(\d+)')
        
        # fetch every line
        matches = pattern.findall(content)
        
        # time: ms 
        layers = []
        for match in matches:
            row_dict = {
                "Name": match[0],
                "Self CPU %": float(match[1][:-1]),
                "Self CPU": self.convert_to_ms(match[2]),
                "CPU total %": float(match[3][:-1]),
                "CPU total": self.convert_to_ms(match[4]),
                "CPU time avg": self.convert_to_ms(match[5]),
                "# of Calls": int(match[6])
            }
            layers.append(row_dict)
        
        return layers

    @property
    def latency(self):
        return self.cpu_time
        # return self.self_cpu_total
    
    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results



class PytorchCPUProfiler(PytorchProfiler):
    use_cuda = False


class PytorchCPUBackend(PytorchBackend):
    parser_class = PytorchCPULatencyParser
    profiler_class = PytorchCPUProfiler

if __name__ == '__main__':
    # use pre-trained resnet18 model as an example
    model = models.resnet18()
    weight_path = '/mnt/sda/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    
    profiler = PytorchCPUProfiler(model, weight_path, data_type='float32')
    shapes = [[1, 3, 224, 224]]
    output = profiler.profile(shapes)
    print(output)
    
    parser = PytorchCPULatencyParser()
    parser.parse(output)
    print(parser.cpu_total_avg)
    print(parser.runs)
    print(parser.self_cpu_total)
    print(parser.layers)
