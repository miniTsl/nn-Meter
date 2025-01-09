import logging
from nn_meter.builder.backends import BaseBackend
import torch

logging = logging.getLogger("nn-Meter")


class PytorchBackend(BaseBackend):
    parser_class = None
    profiler_class = None

    def update_configs(self):
        """update the config parameters for Pytorch platform (CPU or GPU)
        """
        super().update_configs()
        
        self.profiler_kwargs.update({
            'num_runs': self.configs['NUM_RUNS'],
            'warm_ups': self.configs['WARM_UPS']
        })
        
    def test_connection(self):
        """check the status of backend interface connection
        """
        is_cuda_available = torch.cuda.is_available()
        logging.keyinfo("Hello Pytorch CPU backend!")
        if is_cuda_available:
            logging.keyinfo("Hello Pytorch CUDA backend!")
        else:
            logging.keyinfo("Pytorch CUDA backend is not available.")
        
    def profile(self, model_path, metrics = ['latency'], input_shape = None, **kwargs):
        return self.parser.parse(self.profiler.profile(model_path, input_shape, **kwargs)).results.get(metrics)

    # no need to test connection because code will be run on local machine with NVIDIA GPU