
def transfer_learning_suite():
    import subprocess
    import time
    import psutil

    # batch_sizes = [1]
    batch_sizes = [32]
    learning_rates = [0.00001] # remove 0.001

    optimizers = ['adam']
    # pretrained_models = ['Xception', 'VGG16', 'ResNet152V2', 'InceptionResNetV2', 'MobileNetV2']
    pretrained_models = ['InceptionResNetV2']
    # pretrained_models = ['InceptionResNetV2']
    # epochs = [50, 100]
    epochs = [50, 100]
    dropout_rates = [0.3, 0.6, 0.8]
    # dropout_rates = [0]
    activation_functions = ['relu'] # remove softmax
    percentages = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # percentages = [1]
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
                                    ps_pid = subprocess.Popen(["python", "derm_percentage_test.py" ,
                                        model, str(batch_size), optimizer, str(learning_rate),
                                        str(epoch), str(0), activation_function, 
                                        str(dropout_rate), str(percentage)])
                                    while ps_pid.poll() is None:
                                        time.sleep(0.01)


if __name__ == '__main__':
    transfer_learning_suite()

# nået til den her:
# pret model 	Xception
# batch size 	32
# optimizer 	adam
# learning r 	0.0001
# t epochs 	50
# momentum 	0.0
# acti func 	relu
# dropout r 	0.3
# percentage 	0.45
# (ikke inkluderet)