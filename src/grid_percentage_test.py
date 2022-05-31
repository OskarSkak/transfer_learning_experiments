
def transfer_learning_suite():
    import subprocess
    import time
    import psutil

    batch_sizes = [32]
    # batch_sizes = [32, 64]
    learning_rates = [0.00001]# [0.00001, 0.0001] # remove 0.001

    optimizers = ['adam']
    # pretrained_models = ['Xception', 'VGG16', 'ResNet152V2', 'InceptionResNetV2', 'MobileNetV2']
    pretrained_models = ['Xception']
    # pretrained_models = ['InceptionResNetV2']
    # epochs = [50, 100]
    epochs = [100]# [50, 100]
    dropout_rates = [0]# [0, 0.3]
    # dropout_rates = [0]
    activation_functions = ['relu'] # remove softmax
    percentages = [0.75]
    #percentages = # [0.15, 0.25, 0.35, 0.45]
    # percentages = [0.55, 0.65, 0.75, 0.85, 0.95]
    for percentage in percentages:
        for model in pretrained_models:
            for batch_size in batch_sizes:
                for optimizer in optimizers:
                    for learning_rate in learning_rates:
                        for epoch in epochs:
                            for activation_function in activation_functions:
                                for dropout_rate in dropout_rates:
                                    ps_pid = subprocess.Popen(["python", "transfer_learning_percentage_test.py" ,
                                        model, str(batch_size), optimizer, str(learning_rate),
                                        str(epoch), str(0), activation_function, 
                                        str(dropout_rate), str(percentage)])
                                    while ps_pid.poll() is None:
                                        time.sleep(0.01)


if __name__ == '__main__':
    transfer_learning_suite()