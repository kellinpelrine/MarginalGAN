import logging

import numpy as np
from sklearn import linear_model, preprocessing, pipeline
import matplotlib
matplotlib.use('Agg')
from pylab import plt
from matplotlib import pyplot

from competition import AdversarialCompetition
from models import GenerativeNormalModel, GenerativeNormalMixtureModel, GenerativeNormalQuasiMixtureModel
from gradient_descent import GradientDescent

pyplot.ioff()

logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)15s() %(asctime)-15s ] %(message)s", level=logging.DEBUG)

np.random.seed(0)
size_batch = 100
mu_param = 3
sigma_param = 1.0
sample_size = 1000
true_data1 = GenerativeNormalModel(mu_param,sigma_param).random(sample_size).reshape(-1)
true_data2 = GenerativeNormalModel(-mu_param,sigma_param).random(sample_size).reshape(-1)
true_data = np.concatenate((true_data1,true_data2))

competition1 = AdversarialCompetition(
    size_batch=size_batch,
    true_model=GenerativeNormalModel(mu_param, sigma_param),
    discriminative=pipeline.make_pipeline(
        preprocessing.PolynomialFeatures(4),
        linear_model.LogisticRegression()),
    generative=GenerativeNormalModel(
        2.5,2.1,updates=["mu", "sigma"]),
    gradient_descent=GradientDescent(
        0.03, inertia=0.0, annealing=100),
    x_dataset = true_data1
)

competition2 = AdversarialCompetition(
    size_batch=size_batch,
    true_model=GenerativeNormalModel(-mu_param, sigma_param),
    discriminative=pipeline.make_pipeline(
        preprocessing.PolynomialFeatures(4),
        linear_model.LogisticRegression()),
    generative=GenerativeNormalModel(
        -2.5,2.1,updates=["mu", "sigma"]),
    gradient_descent=GradientDescent(
        0.03, inertia=0.0, annealing=100),
    x_dataset = true_data2
)


competition = AdversarialCompetition(
    size_batch=size_batch,
    true_model=GenerativeNormalQuasiMixtureModel(mu_param, sigma_param),
    discriminative=pipeline.make_pipeline(
        preprocessing.PolynomialFeatures(4),
        linear_model.LogisticRegression()),
    generative=GenerativeNormalQuasiMixtureModel(
        2.5,2.1,updates=["mu", "sigma"]),
    gradient_descent=GradientDescent(
        0.03, inertia=0.0, annealing=100),
    x_dataset = true_data
)

print(competition)

for i in range(1001):
    if i % 50 == 0:
        plt.figure()
        competition.plot()   
        pyplot.savefig('Pooling.png')
        pyplot.close()
        plt.figure()
        competition1.plot()
        competition2.plot()
        pyplot.savefig('Separating.png')
        pyplot.close()
        pass
    competition.iteration()
    competition1.iteration()
    competition2.iteration()

print("final model pooling %s" % competition.generatives[-1])
print("final model separating %s" % competition1.generatives[-1], competition2.generatives[-1])

#separated = GenerativeNormalMixtureModel([competition1.generatives[-1].params["mu"], competition2.generatives[-1].params["mu"]], [competition1.generatives[-1].params["sigma"], competition2.generatives[-1].params["sigma"]])
separated1 = GenerativeNormalModel(competition1.generatives[-1].params["mu"], competition1.generatives[-1].params["sigma"])
separated2 = GenerativeNormalModel(competition2.generatives[-1].params["mu"], competition2.generatives[-1].params["sigma"])
pooled = GenerativeNormalQuasiMixtureModel(competition.generatives[-1].params["mu"], competition.generatives[-1].params["sigma"])
#true_mixture_model = GenerativeNormalMixtureModel([mu_param, -mu_param], [sigma_param, sigma_param])
true_mixture_model2 = GenerativeNormalQuasiMixtureModel(mu_param,sigma_param)

plt.figure()
xplot = np.arange(-10, 10, 0.1).reshape((-1, 1))
#plt.plot(xplot, true_mixture_model.predict_proba(xplot), c="black")
plt.plot(xplot, true_mixture_model2.predict_proba(xplot), c="black")
plt.plot(xplot, pooled.predict_proba(xplot), c="red")
plt.plot(xplot, ( separated1.predict_proba(xplot) + separated2.predict_proba(xplot) ) / 2, c="blue")
pyplot.savefig('Combined.png')
pyplot.close()

'''
plt.figure()
competition.plot_params()
#plt.show()
pyplot.savefig('file2.png')
pyplot.close()

plt.figure()
competition.plot_auc()
#plt.show()
pyplot.savefig('file3.png')
pyplot.close()
'''
