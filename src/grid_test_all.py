def transfer_learning_suite():
    import subprocess
    import time
    import psutil

    # batch_sizes = [64]
    # learning_rates = [0.000011]
    # optimizers = ['adam'] 
    # pretrained_models = ['Xception', 'VGG16', 'ResNet152V2', 'InceptionResNetV2', 'MobileNetV2']
    # epochs = [1]
    # dropout_rates = [0.2111] #HUSK
    # activation_functions = ['relu']
    # for model in pretrained_models:
    #     for batch_size in batch_sizes:
    #         for optimizer in optimizers:
    #             for learning_rate in learning_rates:
    #                 for epoch in epochs:
    #                     for activation_function in activation_functions:
    #                         for dropout_rate in dropout_rates:
    #                             ps_pid = subprocess.Popen(["python", "transfer_learning_suite.py" ,
    #                                 model, str(batch_size), optimizer, str(learning_rate),
    #                                 str(epoch), str(0), activation_function, 
    #                                 str(dropout_rate)])
    #                             while ps_pid.poll() is None:
    #                                 time.sleep(0.01)
    # batch_sizes = [64]
    # learning_rates = [0.00001]
    # optimizers = ['adam']
    # pretrained_models = ['Xception', 'VGG16', 'ResNet152V2', 'InceptionResNetV2', 'MobileNetV2']
    # epochs = [1]
    # dropout_rates = [0.211]
    # activation_functions = ['relu']
    # for model in pretrained_models:
    #     for batch_size in batch_sizes:
    #         for optimizer in optimizers:
    #             for learning_rate in learning_rates:
    #                 for epoch in epochs:
    #                     for activation_function in activation_functions:
    #                         for dropout_rate in dropout_rates:
    #                             ps_pid = subprocess.Popen(["python", "derm_percentage_test.py" ,
    #                                 model, str(batch_size), optimizer, str(learning_rate),
    #                                 str(epoch), str(0), activation_function, 
    #                                 str(dropout_rate), str(1)])
    #                             while ps_pid.poll() is None:
    #                                 time.sleep(0.01)
    
    # batch_sizes = [64]
    # learning_rates = [0.001]

    # optimizers = ['adam']
    # architectures = ['one_vgg', 'two_vgg', 'three_vgg']
    # epochs = [1]
    # dropout_rates = [0.211]
    # activation_functions = ['relu']
    # for model in architectures:
    #     for batch_size in batch_sizes:
    #         for optimizer in optimizers:
    #             for learning_rate in learning_rates:
    #                 for epoch in epochs:
    #                     for activation_function in activation_functions:
    #                         for dropout_rate in dropout_rates:
    #                             ps_pid = subprocess.Popen(["python", "standalone_model.py" ,
    #                                 model, str(batch_size), optimizer, str(learning_rate),
    #                                 str(epoch), str(0), activation_function, 
    #                                 str(dropout_rate)])
    #                             while ps_pid.poll() is None:
    #                                 time.sleep(0.01)
    
    batch_sizes = [64]
    learning_rates = [0.001]

    optimizers = ['adam']
    architectures = ['one_vgg', 'two_vgg', 'three_vgg']
    epochs = [1]
    dropout_rates = [0.211]
    activation_functions = ['relu']
    percentages = [1]
    for percentage in percentages:
        for model in architectures:
            for batch_size in batch_sizes:
                for optimizer in optimizers:
                    for learning_rate in learning_rates:
                        for epoch in epochs:
                            for activation_function in activation_functions:
                                for dropout_rate in dropout_rates:
                                    print(percentage)
                                    ps_pid = subprocess.Popen(["python", "standalone_model_derm.py" ,
                                        model, str(batch_size), optimizer, str(learning_rate),
                                        str(epoch), str(0), activation_function, 
                                        str(dropout_rate), str(percentage)])
                                    while ps_pid.poll() is None:
                                        time.sleep(0.01)

if __name__ == '__main__':
    transfer_learning_suite()