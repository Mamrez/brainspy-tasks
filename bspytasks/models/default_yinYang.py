import torch

from brainspy.processors.dnpu import DNPU
from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils


class SingleDNPUCustomModel(torch.nn.Module):
    def __init__(self, configs):
        super(SingleDNPUCustomModel, self).__init__()
        self.gamma = 1
        self.node_no = 1
        model_data = torch.load(configs['model_dir'],
                                map_location=TorchUtils.get_device())
        processor = Processor(configs, model_data['info'],
                              model_data['model_state_dict'])
        self.dnpu = DNPU(processor=processor,
                         data_input_indices=[[0,6]] *
                         self.node_no,
                         forward_pass_type='vec')
        # Remember to add an input transformation, if required   
        # In this case, the example assumes that input data will be in a range from 0 to 1
        self.dnpu.add_input_transform([-1, 1])

    def forward(self, x):
        x = self.dnpu(x)
        return x

    # If you want to swap from simulation to hardware, or vice-versa you need these functions
    def hw_eval(self, configs, info=None):
        self.eval()
        self.dnpu.hw_eval(configs, info)

    def sw_train(self, configs, info=None, model_state_dict=None):
        self.train()
        self.dnpu.sw_train(configs, info, model_state_dict)

    ##########################################################################################

    # If you want to be able to get information about the ranges from outside, you have to add the following functions.
    def get_input_ranges(self):
        return self.dnpu.get_input_ranges()

    def get_control_ranges(self):
        return self.dnpu.get_control_ranges()

    def get_control_voltages(self):
        return self.dnpu.get_control_voltages()

    def set_control_voltages(self, control_voltages):
        self.dnpu.set_control_voltages(control_voltages)

    def get_clipping_value(self):
        return self.dnpu.get_clipping_value()

    # For being able to maintain control voltages within ranges, you should implement the following functions (only those which you are planning to use)
    def regularizer(self):
        return self.gamma * (self.dnpu.regularizer())

    def constraint_weights(self):
        self.dnpu.constraint_control_voltages()

    # For being able to produce same target size as outputs when using hardware validation
    def format_targets(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnpu.format_targets(x)

    ######################################################################################################################################################

    # If you want to implement on-chip GA, you need these functions
    def is_hardware(self):
        return self.dnpu.processor.is_hardware

    def close(self):
        self.dnpu.close()