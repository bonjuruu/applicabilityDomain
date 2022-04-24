"""
This module implements the Carlini and Wagner L2 attack.
"""
import datetime
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# np.inf does NOT play well with pytorch. 1e10 was used in carlini's implementation
INF = 1e10


def predict(model, X, batch_size, device):
    model.eval()
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tensor_pred = -torch.ones(len(X), dtype=torch.long)

    start = 0
    with torch.no_grad():
        for x in loader:
            x = x[0].to(device)
            n = x.size(0)
            end = start + n
            outputs = model(x)
            tensor_pred[start:end] = outputs.max(1)[1].type(torch.long)
            start += n

    return tensor_pred


class CarliniWagnerAttackL2:
    """
    Class preforms Carlini and Wagner L2 attacks.
    """

    def __init__(self,
                 model,
                 n_classes=10,
                 targeted=False,
                 lr=5e-3,
                 binary_search_steps=5,
                 max_iter=1000,
                 confidence=0.0,
                 initial_const=1e-2,
                 abort_early=True,
                 batch_size=32,
                 clip_values=(0.0, 1.0),
                 check_prob=True,
                 verbose=True):
        """
        Create an instance of Carlini and Wagner L2-norm attack Container.

        targeted : bool
            Should we target one specific class? or just be wrong?
        lr : float
            Larger values converge faster to less accurate results
        binary_search_steps : int
            Number of times to adjust the constant with binary search.
        max_iter : int
            Number of iterations to perform gradient descent
        confidence : float
            How strong the adversarial example should be. The parameter kappa in the paper
        initial_const : float
            The initial constant c_multiplier to pick as a first guess
        abort_early : bool
            If we stop improving, abort gradient descent early
        batch_size : int
            The size of a mini-batch.
        clip_values : tuple
            The clipping lower bound and upper bound for adversarial examples.
        check_prob : bool
            The score should not use softmax probability! Turn is off, if you know what you are doing.
        """
        self.model = model
        self.n_classes = n_classes
        self.targeted = targeted
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.confidence = confidence
        self.initial_const = initial_const
        self.abort_early = abort_early
        self.batch_size = batch_size
        self.clip_values = clip_values
        self.check_prob = check_prob
        self.verbose = verbose

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def generate(self, x, targets=None):
        """
        Generate adversarial examples.

        Parameters
        ----------
        x : numpy.ndarray
            The data for generating adversarial examples. If this parameter is not null, 
            `count` and `use_testset` will be ignored.
        targets : numpy.ndarray, optional
            The expected labels for targeted attack.

        Returns
        -------
        adv : numpy.ndarray
            The adversarial examples which have same shape as x.
        """
        time_start = time.time()
        adv = self._generate(x, targets)
        time_elapsed = time.time() - time_start
        if self.verbose:
            print('Time to complete training {} adv. examples: {}'.format(
                len(x), str(datetime.timedelta(seconds=time_elapsed))))
        return adv

    def _generate(self, X, targets=None):
        repeat = self.binary_search_steps >= 10

        n = len(X)
        # prepare data
        tensor_X = torch.from_numpy(X).type(torch.float32)
        tensor_labels = predict(self.model,
                                tensor_X,
                                self.batch_size,
                                self.device)
        dataset = TensorDataset(tensor_X, tensor_labels)
        loader = DataLoader(dataset,
                            self.batch_size,
                            shuffle=False)

        full_input_np = np.zeros_like(X, dtype=np.float32)
        full_adv_np = np.zeros_like(X, dtype=np.float32)
        full_l2_np = 1e9 * np.ones(n, dtype=np.float32)
        full_label_np = -np.ones(n, dtype=np.int64)
        full_pred_np = -np.ones(n, dtype=np.int64)

        count = 0  # only count the image can be classified correct
        for x, y in tqdm(loader, disable=(not self.verbose)):
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = len(x)

            # c is the lagrange multiplier for optimization objective
            lower_bounds_np = np.zeros(batch_size, dtype=np.float32)
            c_np = np.ones(batch_size, dtype=np.float32) * self.initial_const
            upper_bounds_np = np.ones(batch_size, dtype=np.float32) * 1e10

            # overall results
            o_best_l2_np = np.ones(batch_size, dtype=np.float32) * INF
            o_best_pred_np = -np.ones(batch_size, dtype=np.int64)
            o_best_adv = x.detach().clone()  # uses same device as x

            # we optimize over the tanh-space
            x_tanh = self.__to_tanh(x, self.device)

            # NOTE: testing untargeted attack here!
            targets = y
            # y in one-hot encoding
            targets_oh = self.__onehot_encoding(targets)

            # the perturbation variable to optimize (In Carlini's code it's denoted as `modifier`)
            pert_tanh = torch.zeros_like(
                x, requires_grad=True)  # uses same device as x

            # we retrain it for every batch
            optimizer = torch.optim.Adam([pert_tanh], lr=self.lr)

            for sstep in range(self.binary_search_steps):
                # at least try upper bound once
                if repeat and sstep == self.binary_search_steps - 1:
                    c_np = upper_bounds_np

                c = torch.from_numpy(c_np)
                c = c.to(self.device)

                best_l2_np = np.ones(batch_size, dtype=np.float32) * INF
                best_pred_np = -np.ones(batch_size, dtype=np.int64)

                # previous (summed) batch loss, to be used in early stopping policy
                prev_batch_loss = INF  # type: float

                # optimization step
                for ostep in range(self.max_iter):
                    loss, l2_norms, adv_outputs, advs = self.__optimize(
                        optimizer, x_tanh, pert_tanh, targets_oh, c)

                    # check if we should abort search if we're getting nowhere
                    if self.abort_early and ostep % (self.max_iter // 10) == 0:
                        loss = loss.cpu().detach().item()
                        if loss > prev_batch_loss * (1 - 1e-4):
                            break
                        prev_batch_loss = loss  # only check it 10 times

                    # update result
                    adv_outputs_np = adv_outputs.cpu().detach().numpy()
                    targets_np = targets.cpu().detach().numpy()

                    # compensate outputs with parameter confidence
                    adv_outputs_np = self.__compensate_confidence(
                        adv_outputs_np, targets_np)
                    adv_predictions_np = np.argmax(adv_outputs_np, axis=1)

                    for i in range(batch_size):
                        i_l2 = l2_norms[i].item()
                        i_adv_pred = adv_predictions_np[i]
                        i_target = targets_np[i]
                        i_adv = advs[i]  # a tensor

                        if self.__does_attack_success(i_adv_pred, i_target):
                            if i_l2 < best_l2_np[i]:
                                best_l2_np[i] = i_l2
                                best_pred_np[i] = i_adv_pred

                            if i_l2 < o_best_l2_np[i]:
                                o_best_l2_np[i] = i_l2
                                o_best_pred_np[i] = i_adv_pred
                                o_best_adv[i] = i_adv

                # binary search for c
                for i in range(batch_size):
                    if best_pred_np[i] != -1:  # successful, try lower `c` value
                        # update upper bound, and divide c by 2
                        upper_bounds_np[i] = min(upper_bounds_np[i], c_np[i])
                        # 1e9 was used in carlini's implementation
                        if upper_bounds_np[i] < 1e9:
                            c_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2.

                    else:  # failure, try larger `c` value
                        # either multiply by 10 if no solution found yet
                        # or do binary search with the known upper bound
                        lower_bounds_np[i] = max(lower_bounds_np[i], c_np[i])

                        # 1e9 was used in carlini's implementation
                        if upper_bounds_np[i] < 1e9:
                            c_np[i] = (lower_bounds_np[i] +
                                       upper_bounds_np[i]) / 2.
                        else:
                            c_np[i] *= 10

            # save results
            full_l2_np[count: count + batch_size] = o_best_l2_np
            full_label_np[count: count + batch_size] = y.cpu().detach().numpy()
            full_pred_np[count: count + batch_size] = o_best_pred_np
            full_input_np[count: count + batch_size] = x.cpu().detach().numpy()
            full_adv_np[count: count + batch_size] = o_best_adv.cpu().detach().numpy()
            count += batch_size
        return full_adv_np

    @staticmethod
    def __arctanh(x, epsilon=1e-6):
        # to enhance numeric stability. avoiding divide by zero
        x = x * (1 - epsilon)
        return 0.5 * torch.log((1. + x) / (1. - x))

    def __to_tanh(self, x, device=None):
        dmin = torch.tensor(self.clip_values[0], dtype=torch.float32)
        dmax = torch.tensor(self.clip_values[1], dtype=torch.float32)
        if device is not None:
            dmin = dmin.to(device)
            dmax = dmax.to(device)
        box_mul = (dmax - dmin) * .5
        box_plus = (dmax + dmin) * .5
        return self.__arctanh((x - box_plus) / box_mul)

    def __from_tanh(self, w, device=None):
        dmin = torch.tensor(self.clip_values[0], dtype=torch.float32)
        dmax = torch.tensor(self.clip_values[1], dtype=torch.float32)
        if device is not None:
            dmin = dmin.to(device)
            dmax = dmax.to(device)
        box_mul = (dmax - dmin) * .5
        box_plus = (dmax + dmin) * .5
        return torch.tanh(w) * box_mul + box_plus

    def __onehot_encoding(self, labels):
        labels_t = labels.unsqueeze(1)
        y_onehot = torch.zeros(len(labels), self.n_classes, dtype=torch.int8)
        y_onehot = y_onehot.to(self.device)
        return y_onehot.scatter_(1, labels_t, 1)

    @staticmethod
    def __get_l2_norm(a, b, dim=1):
        return torch.norm(
            a.view(a.size(0), -1) - b.view(b.size(0), -1),
            dim=dim)

    def __optimize(self, optimizer, inputs_tanh, pert_tanh, targets_oh, const):
        batch_size = inputs_tanh.size(0)
        is_targeted = self.targeted

        optimizer.zero_grad()
        # the adversarial examples in image space
        advs = self.__from_tanh(inputs_tanh + pert_tanh, self.device)
        # the clean images converted back from tanh space
        inputs = self.__from_tanh(inputs_tanh, self.device)

        # The softmax is stripped out from this model.
        adv_outputs = self.model(advs)

        if self.check_prob and torch.equal(
                torch.round(adv_outputs.sum(1)),
                torch.ones(len(adv_outputs)).to(self.device)):
            raise ValueError('The score from the model should NOT be probability!')

        l2_norms = self.__get_l2_norm(advs, inputs)
        assert l2_norms.size() == (batch_size,)

        target_outputs = torch.sum(targets_oh * adv_outputs, 1)
        other_outputs = torch.max(
            (1.0 - targets_oh) * adv_outputs - targets_oh * INF, 1)[0]

        if is_targeted:
            f_loss = torch.clamp(other_outputs - target_outputs + self.confidence, min=0.)
        else:
            f_loss = torch.clamp(target_outputs - other_outputs + self.confidence, min=0.)

        loss = torch.sum(l2_norms + const * f_loss)
        loss.backward()
        optimizer.step()

        return loss, l2_norms, adv_outputs, advs

    def __compensate_confidence(self, outputs, targets):
        is_targeted = self.targeted

        outputs_comp = np.copy(outputs)
        indices = np.arange(targets.shape[0])
        if is_targeted:
            outputs_comp[indices, targets] -= self.confidence
        else:
            outputs_comp[indices, targets] += self.confidence

        return outputs_comp

    def __does_attack_success(self, pred, label):
        if self.targeted:
            return int(pred) == int(label)  # match the target label
        else:
            return int(pred) != int(label)  # anyting other than the true label
