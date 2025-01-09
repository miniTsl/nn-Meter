# import torch  # This is all you need to use both PyTorch and TorchScript!
# print(torch.__version__)
# torch.manual_seed(191009)  # set the seed for reproducibility
#
# class MyDecisionGate(torch.nn.Module):
#     def forward(self, x):
#         if x.sum() > 0:
#             return x
#         else:
#             return -x
#
# class MyCell(torch.nn.Module):
#     def __init__(self):
#         super(MyCell, self).__init__()
#         self.dg = MyDecisionGate()
#         self.linear = torch.nn.Linear(4, 4)
#
#     def forward(self, x, h):
#         new_h = torch.tanh(self.dg(self.linear(x)) + h)
#         return new_h, new_h
#
# my_cell = MyCell()
# x = torch.rand(3, 4)
# h = torch.rand(3, 4)
# print(my_cell)
# print(my_cell(x, h))
#
# # using jit.trace to trace the cell
# traced_cell = torch.jit.trace(my_cell, (x, h))  # Trace a function and return an executable or :class:`ScriptFunction` that will be optimized using just-in-time compilation.
# # things like control flow are erased. The trace only shows the computation that actually happened.
# print(traced_cell.dg.code)
# print(traced_cell.code)
# # trace can be used to infer different shapes of input tensors from the example input tensors while tracing.
# print(traced_cell(x[0:1], h[0:1]))
#
# # using jit.script to compile the cell
# scripted_gate = torch.jit.script(MyDecisionGate())  # Scripting a function or ``nn. Module`` will inspect the source code, compile it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or :class:`ScriptFunction`.
# # the scripted_gate is a torchscript object that can be used like a regular PyTorch module, maintaining the control flow.
# print(scripted_gate.code)
# scripted_cell = torch.jit.script(my_cell)
# print(scripted_cell.dg.code)
# print(scripted_cell.code)
# print(scripted_cell(x, h))
# # the scripted_cell can also be used with different shapes of input tensors.
# print(scripted_cell(x[0:1], h[0:1]))
#
#
# # save the traced model to a file
# traced_cell.save("traced_cell.pt")
# # load the model back
# loaded_cell = torch.jit.load("traced_cell.pt")
# print(loaded_cell)
# # inference with the loaded model
# print(loaded_cell(x[0:1], h[0:1]))
#
# # or save the scripted model to a file
# scripted_cell.save("scripted_cell.pt")
# # load the model back
# loaded_cell = torch.jit.load("scripted_cell.pt")
# print(loaded_cell)
# # inference with the loaded model
# print(loaded_cell(x[0:1], h[0:1]))

# examine the graph or code of the loaded model
workspace = "/mnt/sda/nn-Meter/workspaces/pytorch_cpu"
path = workspace+"/fusion_rule_test/testcases/BF_conv_add_conv.pt"
import torch
model = torch.jit.load(path)
print(model.graph)