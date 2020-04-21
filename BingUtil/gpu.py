import gpustat
import os
import numpy as np
def select_free_gpu():
    mem = []
    gpus = list(set([0,1,2,3,4,5,6,7]))
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    return str(gpus[np.argmin(mem)])
# from within main code before calling anything GPU related. (Check how this is used in rrn.py)
gpu = select_free_gpu()
print("gpu is: ", gpu)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) # this line assigns the GPU to the code