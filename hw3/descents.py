from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE, delta: float = 1.0):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function
        self.delta = delta
        self.dimension = dimension

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        residuals = x @ self.w - y
        if self.loss_function is LossFunction.MSE:
            return np.dot(residuals, residuals) / x.shape[0]
        elif self.loss_function is LossFunction.LogCosh:
            return np.log(np.cosh(residuals)).mean()
        elif self.loss_function is LossFunction.MAE:
            return np.abs(residuals).mean()
        elif self.loss_function is LossFunction.Huber:
            loss = 0
            mse_mask = 1 * (np.abs(residuals) <= self.delta)
            loss += np.dot(mse_mask * residuals, mse_mask * residuals) / 2
            loss += self.delta * np.sum(np.abs((1 - mse_mask) * residuals)) - self.delta **2 / 2
            loss /= x.shape[0]
            return loss
            
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        upd = -self.lr() * gradient
        self.w += upd
        return upd
    
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function is LossFunction.MSE:
            return 2 * x.T @ (x @ self.w - y) / x.shape[0]
        
        elif self.loss_function is LossFunction.LogCosh:
            return x.T @ np.tanh(x @ self.w - y) / x.shape[0]
        
        elif self.loss_function is LossFunction.MAE:
            return x.T @ np.sign(x @ self.w - y) / x.shape[0]
        
        elif self.loss_function is LossFunction.Huber:
            
            grad = np.zeros(self.dimension)
            mse_mask = 1 * (np.abs(y - x @ self.w) <= self.delta)
            grad += x.T @ (mse_mask *(y - x @ self.w)) / x.shape[0]
            grad += self.delta * x.T @ np.sign(((1-mse_mask)*(y - x @ self.w)))/ x.shape[0]

            return grad
    
class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch = np.random.randint(0, x.shape[0], self.batch_size)
        return super().calc_gradient(x[batch], y[batch])

        

class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        
        self.h = self.alpha * self.h + self.lr() * gradient
        self.w -= self.h
        return -self.h

class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        self.iteration += 1
        self.m = self.beta_1 * self.m + (1-self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1-self.beta_2) * gradient**2
        m_hat = self.m/(1-self.beta_1**self.iteration)
        v_hat = self.v/(1-self.beta_2**self.iteration)
        upd = -self.lr() * m_hat / (np.sqrt(v_hat) + self.eps)
        self.w += upd
        return upd


class Adamax(VanillaGradientDescent):
    """
    Adaptive Moment Estimation L-inf gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.u: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.8
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        self.iteration += 1
        self.m = self.beta_1 * self.m + (1-self.beta_1) * gradient
        self.u = np.maximum(self.beta_2 * self.u, np.abs(gradient))
        upd = -self.lr() * self.m / (self.u + self.eps)
        self.w += upd
        return upd
    

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w  
        mask = np.ones(self.w.shape)
        mask[-1] = 0
        return (super().calc_gradient(x, y) + l2_gradient * self.mu) * mask


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'adamax': Adamax if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
