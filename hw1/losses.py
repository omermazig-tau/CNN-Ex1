import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        s_y_i = x_scores.gather(1, y.reshape(-1, 1))
        M = self.delta + x_scores - s_y_i
        L_i = torch.max(M, torch.zeros_like(M))
        L_sum = torch.sum(L_i) - y.size(dim=0) * self.delta
        loss = (L_sum / y.size(dim=0))
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["M"] = M
        self.grad_ctx["x"] = x
        self.grad_ctx["y"] = y
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        M = self.grad_ctx["M"]
        x = self.grad_ctx["x"]
        y = self.grad_ctx["y"]
        G = (M > 0).float()
        G.index_put_((torch.arange(0, len(y)), y), torch.zeros(len(y)))   # if y_i is the column number we will zero the value
        sum_rows = -torch.sum(G, 1)
        G.index_put_((torch.arange(0, len(y)), y), sum_rows) # if y_i is the column number we will put the sum of the row
        grad = x.T @ G
        grad = grad * (1 / y.size(dim=0))
        # ========================

        return grad
