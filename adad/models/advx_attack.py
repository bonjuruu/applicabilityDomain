"""
Perform adversarial attacks.
"""
from pyexpat import model
import numpy as np

from art.attacks.evasion import (AutoProjectedGradientDescent,
                                 FastGradientMethod, CarliniL2Method)
from art.estimators.classification import PyTorchClassifier

from adad.models.attacks.carlini import CarliniWagnerAttackL2


class AdvxAttack:
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 n_features,
                 n_classes,
                 att_name,
                 device,
                 clip_values=(0.0, 1.0),
                 batch_size=128):

        self.clf = PyTorchClassifier(
            model=model,
            loss=loss_fn,
            input_shape=(n_features,),
            optimizer=optimizer,
            nb_classes=n_classes,
            clip_values=clip_values,
            device_type=device
        )
        self.clip_values = clip_values
        self.n_classes = n_classes
        self.name = att_name
        self.batch_size = batch_size

    def generate(self, X, epsilon=0.3, verbose=False):
        X = np.float32(X)
        eps_step = epsilon / 10.0 if epsilon <= 0.1 else 0.1
        if self.name == 'apgd':
            attack = AutoProjectedGradientDescent(
                estimator=self.clf,
                eps=epsilon,
                eps_step=eps_step,
                max_iter=1000,
                targeted=False,
                batch_size=self.batch_size,
                verbose=verbose)
        elif self.name == 'fgsm':
            attack = FastGradientMethod(
                estimator=self.clf,
                eps=epsilon,
                batch_size=self.batch_size)
        elif self.name == 'cw2':
            attack = CarliniWagnerAttackL2(
                model=self.clf._model._model,
                n_classes=self.n_classes,
                confidence=epsilon,
                clip_values=self.clip_values,
                binary_search_steps=5,
                max_iter=100,
                check_prob=False,
                verbose=False)
        adv = attack.generate(x=X)
        return adv
