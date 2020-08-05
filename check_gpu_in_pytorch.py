import torch
print("pytorch version: ", torch.__version__)
print("current device: ", torch.cuda.current_device())
print("device count: ", torch.cuda.device_count())
print("device name: ", torch.cuda.get_device_name(0))
print("device is available: ", torch.cuda.is_available())
