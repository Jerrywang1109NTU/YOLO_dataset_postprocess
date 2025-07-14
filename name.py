import numpy as np

data = np.load('dpu_outputs.npz')
print("Saved arrays:", data.files)

for name in data.files:
    print(f"{name}: shape = {data[name].shape}")