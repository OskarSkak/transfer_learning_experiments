import sys
import time
import subprocess as sp
import os
from threading import Thread , Timer
import sched, time

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

print(sys.argv[0]) # program name
print(sys.argv[1]) # pretrained model
print(sys.argv[2]) # batch sizes
print(sys.argv[3]) # optimizer
print(sys.argv[4]) # learning_rate
print(sys.argv[5]) # epoch
print(sys.argv[6]) # momentum
print(sys.argv[7]) # activation_function
print(sys.argv[8]) # dropout_rate

collecting_gpu = True
gpu_usage = []

def record_gpu():
    if not collecting_gpu:
        return
    Timer(5.0, record_gpu).start()
    gpu_usage.extend(get_gpu_memory())

record_gpu()

a = 100000000
# for aa in range(a):
for b in range(a):
    c = b*2/2+1-4
    c = c*2/2+1

collecting_gpu = False
print(gpu_usage)

time.sleep(30)