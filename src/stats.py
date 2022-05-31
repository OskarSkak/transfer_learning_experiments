import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from db.models import GPUUsageTest, Diagnostic, GPUUsageTrain, get_session, FitHistory
from sqlalchemy import func, or_
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


def get_avg_test_util(res):
    subres = []
    for r in res:
        subres.extend([x.usage for x in r.gpu_train])
    print(f'{res[0].model}: \t{sum(subres)/len(subres)}')

def main():
    s = get_session()

    xceptions = s.query(Diagnostic).filter(Diagnostic.percentage == 1, Diagnostic.dataset == 'derm', Diagnostic.model == 'Xception').all()
    vgg16s = s.query(Diagnostic).filter(Diagnostic.percentage == 1,  Diagnostic.dataset == 'derm', Diagnostic.model == 'VGG16').all()
    resnet = s.query(Diagnostic).filter(Diagnostic.percentage == 1,  Diagnostic.dataset == 'derm', Diagnostic.model == 'ResNet152V2').all()
    inception = s.query(Diagnostic).filter(Diagnostic.percentage == 1, Diagnostic.dataset == 'derm', Diagnostic.model == 'InceptionResNetV2').all()
    mobilenet = s.query(Diagnostic).filter(Diagnostic.percentage == 1, Diagnostic.dataset == 'derm', Diagnostic.model == 'MobileNetV2').all()
    custom = s.query(Diagnostic).filter(Diagnostic.percentage > 1, Diagnostic.model.like('%vgg')).all()

    
    print('ISIC')
    pre_models = ['Xception', 'VGG16', 'ResNet', 'Inception', 'MobileNet', 'Custom']
    results = [xceptions, vgg16s, resnet, inception, mobilenet, custom]

    for r in results:
        get_avg_test_util(r)


    xceptions = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None), Diagnostic.dataset.is_(None), Diagnostic.model == 'Xception').all()
    vgg16s = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None),    Diagnostic.dataset.is_(None), Diagnostic.model == 'VGG16').all()
    resnet = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None),    Diagnostic.dataset.is_(None), Diagnostic.model == 'ResNet152V2').all()
    inception = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None), Diagnostic.dataset.is_(None), Diagnostic.model == 'InceptionResNetV2').all()
    mobilenet = s.query(Diagnostic).filter(Diagnostic.percentage.is_(None), Diagnostic.dataset.is_(None), Diagnostic.model == 'MobileNetV2').all()
    custom = s.query(Diagnostic).filter(Diagnostic.percentage == 0.3,       Diagnostic.dataset.is_(None), Diagnostic.model.like('%vgg')).all()

    pre_models = ['Xception', 'VGG16', 'ResNet', 'Inception', 'MobileNet', 'Custom']
    results = [xceptions, vgg16s, resnet, inception, mobilenet, custom]

    print('RAD Dataset')

    for r in results:
        get_avg_test_util(r)

    
    



if __name__ == '__main__':
    main()