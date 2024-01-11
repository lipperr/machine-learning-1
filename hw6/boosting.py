from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
            blend:bool = False,
            base_model_class = DecisionTreeRegressor
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)    
        
        self.plot: bool = plot
        self.blend = blend

        self.meta_X = None
        self.meta_model = LinearRegression() if self.blend else None

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        
        subsample_idxs = np.random.choice(np.arange(x.shape[0]), np.intc(self.subsample * x.shape[0]))
        
        subsample_x = x[subsample_idxs, :]
        subsample_y = y[subsample_idxs]
        
        target = -self.loss_derivative(subsample_y, predictions[subsample_idxs])
        
        if self.base_model_class is DecisionTreeRegressor:
            model = self.base_model_class(**self.base_model_params).fit(subsample_x, target)
            gamma = self.find_optimal_gamma(y, predictions, model.predict(x))
            self.gammas.append(self.learning_rate * gamma)
        elif self.base_model_class is LogisticRegression:
            model = self.base_model_class().fit(subsample_x, subsample_y)
        
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        if self.blend:
            self.early_stopping_rounds = None
            self.meta_X = np.empty((y_valid.shape[0], self.n_estimators))

        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        
        for i in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions = self.models[-1].predict(x_train)
            valid_predictions = self.models[-1].predict(x_valid)
            
            if self.blend:
                self.meta_X[:, i] = valid_predictions

            if self.plot:
                train_score = self.score(x_train, y_train)
                valid_score = self.score(x_valid, y_valid)
                
                self.history['train'].append(train_score)
                self.history['valid'].append(valid_score)
            
            if self.early_stopping_rounds is not None:
                self.validation_loss[i] = self.loss_fn(y_valid, valid_predictions)
                if i + 1 >= self.early_stopping_rounds and self.validation_loss[i] <= self.validation_loss[i-1]:
                    break
                    
        if self.blend:
            self.meta_model.fit(self.meta_X, y_valid)

        if self.plot:
            sns.lineplot(self.history)
            plt.title('ROC-AUC')

    def predict_proba(self, x):
        if self.blend:
            meta_x = np.empty((x.shape[0], self.n_estimators))
            for i, model in enumerate(self.models):
                if self.base_model_class is DecisionTreeRegressor:
                    meta_x[:, i] = model.predict(x)
                elif self.base_model_class is LogisticRegression:
                    meta_x[:, i] = model.predict_proba(x)[:, 1]

            probs = np.zeros((x.shape[0], 2))
            if self.base_model_class is DecisionTreeRegressor:
                probs[:, 1] = self.sigmoid(self.meta_model.predict(meta_x))
            elif self.base_model_class is LogisticRegression:
                probs[:, 1] = self.meta_model.predict(meta_x)
            probs[:, 0] = 1 - probs[:, 1]
        else: 
            probs = np.zeros((x.shape[0], 2))
            for gamma, model in zip(self.gammas, self.models):
                probs[:, 1] += gamma * model.predict(x)
            probs[:, 1] = self.sigmoid(probs[:, 1])
            probs[:, 0] = 1 - probs[:, 1]
        return probs
        
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        f_imps = np.mean(np.array([model.feature_importances_ for model in self.models]), axis=0)
        # assert np.all(f_imps >= 0)
        # f_imps /= np.sum(f_imps)
        return f_imps
