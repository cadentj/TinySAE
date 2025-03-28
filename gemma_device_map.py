sae_layer = 28
device_split = int(sae_layer * (3/5))
cuda_zero = range(0, device_split)
cuda_one = range(device_split, sae_layer)

device_map = {
    f"model.layers.{i}": "cuda:0" for i in cuda_zero
} |{
    f"model.layers.{i}": "cuda:1" for i in cuda_one
} 