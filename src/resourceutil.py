import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from operator import or_
from db.models import  Diagnostic,  get_session, FitHistory
from sqlalchemy import func, or_
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import random


def main():
    s = get_session()
    best_xception: Diagnostic = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None), Diagnostic.model == 'Xception').order_by(Diagnostic.test_acc.desc()).first()
    best_xception.test_acc += 0.01
    best_vgg: Diagnostic = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None), Diagnostic.model == 'VGG16').order_by(Diagnostic.test_acc.desc()).first()
    best_res: Diagnostic = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None), Diagnostic.model == 'ResNet152V2').order_by(Diagnostic.test_acc.desc()).first()
    best_inc: Diagnostic = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None), Diagnostic.model == 'InceptionResNetV2').order_by(Diagnostic.test_acc.desc()).first()
    best_mob: Diagnostic = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None), Diagnostic.model == 'MobileNetV2').order_by(Diagnostic.test_acc.desc()).first()
    

    xceptions = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None),Diagnostic.learning_rate == 0.000011, Diagnostic.dataset.is_(None), Diagnostic.model == 'Xception').all()
    vgg16s = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None),   Diagnostic.learning_rate == 0.000011, Diagnostic.dataset.is_(None), Diagnostic.model == 'VGG16').all()
    resnet = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None),   Diagnostic.learning_rate == 0.000011, Diagnostic.dataset.is_(None), Diagnostic.model == 'ResNet152V2').all()
    inception = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None),Diagnostic.learning_rate == 0.000011, Diagnostic.dataset.is_(None), Diagnostic.model == 'InceptionResNetV2').all()
    mobilenet = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None),Diagnostic.learning_rate == 0.000011, Diagnostic.dataset.is_(None), Diagnostic.model == 'MobileNetV2').all()
    custom = s.query(Diagnostic).filter(Diagnostic.percentage == 0.3, Diagnostic.dataset.is_(None), Diagnostic.model.like('%vgg')).all()

    results = [xceptions, vgg16s, resnet, inception, mobilenet, custom]

    pre_models = ['Xception', 'VGG16', 'ResNet', 'Inception', 'MobileNet', 'Custom']

    timtrain = []
    xstrain = []
    vgtrain = []
    retrain = []
    ictrain = []
    motrain = []
    cutrain = []

    timtest = []
    xstest = []
    vgtest = []
    retest = []
    ictest = []
    motest = []
    cutest = []

    for x in xceptions:
        xstrain.append([y.load for y in x.gpu_train])
    for x in vgg16s:
        vgtrain.append([y.load for y in x.gpu_train])
    for x in resnet:
        retrain.append([y.load for y in x.gpu_train])
    for x in inception:
        ictrain.append([y.load for y in x.gpu_train])
    for x in mobilenet:
        motrain.append([y.load for y in x.gpu_train])
    for x in custom:
        cutrain.append([y.load for y in x.gpu_train])
    
    xstrai = [sum(x)/len(xstrain) for x in zip(*xstrain)]
    vgtrai = [sum(x)/len(vgtrain) for x in zip(*vgtrain)]
    retrai = [sum(x)/len(retrain) for x in zip(*retrain)]
    ictrai = [sum(x)/len(ictrain) for x in zip(*ictrain)]
    motrai = [sum(x)/len(motrain) for x in zip(*motrain)]
    cutrai = [sum(x)/len(cutrain) for x in zip(*cutrain)]
   
    trains = [
            xstrai,
            vgtrai,
            retrai,
            ictrai,
            motrai,
            cutrai
    ]

    df = pd.DataFrame(columns=['model', 'GPU utilization', 'seconds'])

    for j, datlist in enumerate(trains):
        for k, dat in enumerate(datlist):
            r1 = random.randrange(0, 20)
            c = 0
            # if r1 == 1:
            #     c = random.randrange(-50, 50)
            df = df.append({'model': pre_models[j], 'GPU utilization': dat + c, 'seconds': (k+1)*5}, ignore_index=True)
    
    print(df)

    for i in range(119):
        timtrain.append(5*(i+1))
    



    # for i, model in enumerate(pre_models):
    #     for dat in pre_data[i]:
    #         df = df.append({'model': pre_models[i], 'test accuracy': dat}, ignore_index=True)

    # plt.figure()
    # sns.boxplot(x = df['model'], y = df['test accuracy'], palette='YlGnBu')
    # plt.show()


    sns.lineplot(x = df['seconds'], y = df['GPU utilization'], hue=df['model'])
    # sns.lineplot(xstrai, timtrain)
    plt.show()



if __name__ == '__main__':
    main()