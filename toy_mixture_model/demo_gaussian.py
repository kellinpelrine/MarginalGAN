import logging

import numpy as np
from sklearn import linear_model, preprocessing, pipeline
import matplotlib
matplotlib.use('Agg')
from pylab import plt
from matplotlib import pyplot

from competition import AdversarialCompetition
from models import GenerativeNormalModel
from gradient_descent import GradientDescent

pyplot.ioff()

logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)15s() %(asctime)-15s ] %(message)s", level=logging.DEBUG)

np.random.seed(0)
size_batch = 1000

competition = AdversarialCompetition(
    size_batch=size_batch,
    true_model=GenerativeNormalModel(1, 2),
    discriminative=pipeline.make_pipeline(
        preprocessing.PolynomialFeatures(4),
        linear_model.LogisticRegression()),
    generative=GenerativeNormalModel(
        0, 1, updates=["mu", "sigma"]),
    gradient_descent=GradientDescent(
        0.03, inertia=0.0, annealing=100),
)

print(competition)

for i in range(500):
    if i % 50 == 0:
        competition.plot()   
        pyplot.savefig('file.png')
        pyplot.close()
        pass
    competition.iteration()

print("final model %s" % competition.generatives[-1])

competition.plot_params()
#plt.show()
pyplot.savefig('file2.png')
pyplot.close()

competition.plot_auc()
#plt.show()
pyplot.savefig('file3.png')
pyplot.close()
