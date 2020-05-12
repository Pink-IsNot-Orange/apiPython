from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import matplotlib.pyplot as plt


# VARIABLES (y -> objetivo, x -> imagenes)
y = np.load(r'C:\Users\JONA-PC\Desktop\pythonApi\train_labels.npy')
x = np.load(r'C:\Users\JONA-PC\Desktop\pythonApi\train.npy')
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

imgSet = 7000
# Separacion en grupos de prueba y validacion
x_train = x[:imgSet]
y_train = y[:imgSet]
x_test = x[imgSet+1:]
y_test = y[imgSet+1:]

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(f'Decision Tree Classifier - Entrenada con {imgSet} muestras: {100*round(accuracy_score(y_test, y_pred), 3)}% ')

#CREACION DE LA RED NEURONAL
""" mlp = MLPClassifier(max_iter=200)

parameter_space = {
    'hidden_layer_sizes': [(32, 16 , 8), (64, 32, 16), (128, 64, 32)],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [1e-4, 5e-3],
    'learning_rate': ['constant', 'adaptive'],
}

gsCV = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=10)
gsCV.fit(x_train, y_train)

# Best paramete set
print('Best parameters found:\n', gsCV.best_params_)

# All results
means = gsCV.cv_results_['mean_test_score']
stds = gsCV.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gsCV.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


y_true, y_pred = y_test , gsCV.predict(x_test)


print('Results on the test set:')
print(classification_report(y_true, y_pred)) """
""" mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8, 16), max_iter=600, activation='relu',
                    alpha=1e-4, solver='adam', tol=1e-4, random_state=None,
                    learning_rate='constant', learning_rate_init=0.01, verbose=True)

#ENTRENAMIENTO
nn = mlp.fit(x_train, y_train)
MLPy_pred = nn.predict(x_test)

print(f'MLPClassifier - Entrnada con {imgSet} muestras: {100*round(accuracy_score(y_test, MLPy_pred), 3)}% accuracy') """

""" 
plt.plot(nn.loss_curve_)
plt.title('Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.show() """
# Save with pickle to dir
""" filename = 'finalized_model.sav'
pickle.dump(nn, open(filename, 'wb')) """
