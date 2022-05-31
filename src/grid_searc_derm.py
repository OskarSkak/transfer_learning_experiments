

# Grid search

# Hyperparameters of interest:
# Batch size 1, 5, 20
# learning rate 0.0001, 0.001, 0.01 
# optimizer (SGD, Adam, RMSprop)

def transfer_learning_suite():
    import subprocess
    import time
    import psutil

    # batch_sizes = [32, 64]
    batch_sizes = [64]
    learning_rates = [0.00001, 0.0001] # remove 0.001
    # optimizers = ['sgd', 'adam']
    optimizers = ['adam'] # SGd sucks
    # pretrained_models = ['Xception', 'VGG16', 'ResNet152V2', 'InceptionResNetV2', 'MobileNetV2']
    pretrained_models = ['VGG16', 'ResNet152V2', 'InceptionResNetV2', 'MobileNetV2']
    
    epochs = [50, 100]
    dropout_rates = [0, 0.4]
    activation_functions = ['relu'] # remove softmax
    for model in pretrained_models:
        for batch_size in batch_sizes:
            for optimizer in optimizers:
                for learning_rate in learning_rates:
                    for epoch in epochs:
                        for activation_function in activation_functions:
                            for dropout_rate in dropout_rates:
                                ps_pid = subprocess.Popen(["python", "derm_percentage_test.py" ,
                                    model, str(batch_size), optimizer, str(learning_rate),
                                    str(epoch), str(0), activation_function, 
                                    str(dropout_rate), str(1)])
                                while ps_pid.poll() is None:
                                    time.sleep(0.01)
                    # _cpu = []
                    # _mem = []
                    # start = time.time()

                    # ps_pid = subprocess.Popen(["python", "transfer_learning.py" ,
                    #     model, batch_size, optimizer, learning_rate])
                    # ps_pid = subprocess.Popen(["python", "test.py" ,
                    #     model, str(batch_size), optimizer, str(learning_rate)])
                    # while ps_pid.poll() is None:
                    #     time.sleep(0.01)
                    # print("pid: ", str(ps_pid.pid))
                    # p = psutil.Process(ps_pid.pid)
                    # while ps_pid.poll() is None:
                    #     time.sleep(5)
                    #     # print("CPU: ", p.cpu_percent(interval=None)/12)
                    #     # print("MEM: ", p.memory_percent())
                    #     time.sleep(5)
                    #     usage = str(subprocess.run(["ps", "-o", "pid,%cpu,%mem", "-fp", str(ps_pid.pid)], capture_output=True)).split("\\n")[1]
                    #     cpu = usage.split(" ")[2]
                    #     if cpu == "":
                    #         cpu_u = float(cpu)/12
                    #         mem_u = str(usage).split(" ")[4]
                    #         _cpu.append(cpu_u)
                    #         _mem.append(float(mem_u))
                    # while ps_pid.poll() is None:
                    #     time.sleep(0.01)
                    # dur = time.time() - start
                    # # name = f'{model}_size_{batch_size}_opt_{optimizer}_learningrate_{learning_rate}'
                    # with open(f'{model}.txt', 'a+') as f:
                    #     f.write(f'# model: {model}')
                    #     f.write(f'\n# batch size: {batch_size}')
                    #     f.write(f'\n# optimizer: {optimizer}')
                    #     f.write(f'\n# learning rate: {learning_rate}')
                    #     f.write(f'\n# training time: {dur}')
                    #     f.write(f'\nmemory = {_mem}')
                    #     f.write(f'\ncpu = {_cpu}')
                    #     f.write('\n-------------------------------------------------------------------------------\n')


if __name__ == '__main__':
    transfer_learning_suite()