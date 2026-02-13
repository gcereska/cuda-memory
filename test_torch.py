import torch
print("torch:", torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)      # likely "13.0" 
print(torch.cuda.get_device_name(0))
