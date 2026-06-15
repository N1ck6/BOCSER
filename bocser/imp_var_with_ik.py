import tensorflow as tf
import tensorflow_probability as tfp

from trieste.types import TensorType
from trieste.data import Dataset
from trieste.acquisition.interface import (AcquisitionFunction,
                                           AcquisitionFunctionClass,
                                           SingleModelAcquisitionBuilder)
from trieste.models import ProbabilisticModel

import typing

from ik_loss import IKLoss


class ImprovementVarianceWithIK(SingleModelAcquisitionBuilder):
    """Returns variance of Improvement I(x) = max(eta + threshold - f(x), 0).
    """

    def __init__(
        self,
        threshold: float,
        ik_loss: IKLoss,
        ik_loss_idxs: typing.List[int],
        ik_loss_weight: float = 1.,
    ):

        self._threshold = threshold
        self._ik_loss = ik_loss
        self._ik_loss_weight = ik_loss_weight
        self._ik_loss_idxs = ik_loss_idxs

    def __repr__(self) -> str:
        """"""
        return "ImprovementVariance()"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Dataset = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The improvement variance function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = typing.cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset),
                                     message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return improvement_variance(model=model,
                                    eta=eta,
                                    dataset=dataset,
                                    threshold=self._threshold,
                                    ik_loss=self._ik_loss,
                                    ik_loss_idxs=self._ik_loss_idxs,
                                    ik_loss_weight=self._ik_loss_weight)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Dataset = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.  Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = typing.cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset),
                                     message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, improvement_variance), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        function.update(eta, dataset, self._threshold, self._ik_loss,
                        self._ik_loss_idxs,
                        self._ik_loss_weight)  # type: ignore
        return function


class improvement_variance(AcquisitionFunctionClass):

    def __init__(self,
                 model: ProbabilisticModel,
                 eta: TensorType,
                 dataset: Dataset,
                 threshold: float,
                 ik_loss: IKLoss,
                 ik_loss_idxs: typing.List[int],
                 ik_loss_weight: float = 1.):
        """"""
        self._model = model
        self._eta = tf.Variable(eta)
        self._dataset = dataset
        self._threshold = threshold
        self._ik_loss = ik_loss
        self._ik_loss_idxs = ik_loss_idxs
        self._ik_loss_weight = ik_loss_weight

        if not isinstance(self._ik_loss_weight, tf.Tensor):
            self._ik_loss_weight = tf.constant(self._ik_loss_weight,
                                               dtype=tf.float64)

    def update(self,
               eta: TensorType,
               dataset: Dataset,
               threshold: float,
               ik_loss: IKLoss,
               ik_loss_idxs: typing.List[int],
               ik_loss_weight: float = 1.) -> None:
        """Update the acquisition function with a new eta value, dataset, threshold"""
        self._eta.assign(eta)
        self._dataset = dataset
        self._threshold = threshold
        self._ik_loss = ik_loss
        self._ik_loss_idxs = ik_loss_idxs
        self._ik_loss_weight = ik_loss_weight

        if not isinstance(self._ik_loss_weight, tf.Tensor):
            self._ik_loss_weight = tf.constant(self._ik_loss_weight,
                                               dtype=tf.float64)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:

        mean, variance = self._model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        tau = self._eta + self._threshold

        gathered_result = safe_gather_with_none(x, self._ik_loss_idxs)
        ring_dihedrals = [gathered_result[:, i].to_tensor() for i in range(len(self._ik_loss_idxs))]

        return (normal.cdf(tau) * (((tau - mean)**2) *
                                   (1 - normal.cdf(tau)) + variance) +
                tf.sqrt(variance) * normal.prob(tau) * (tau - mean) *
                (1 - 2 * normal.cdf(tau)) - variance * (normal.prob(tau)**2) -
                self._ik_loss_weight * self._ik_loss(ring_dihedrals))

def safe_gather_with_none(x, indices):
    x = tf.convert_to_tensor(x)
    indices = tf.ragged.constant(indices, dtype=tf.int32)

    nan = tf.constant(float("nan"), x.dtype)

    def gather_one(xb):
        def gather_flat(idx):
            safe_idx = tf.maximum(idx, 0)
            val = tf.gather(xb, safe_idx)
            return tf.where(idx > -1, val, nan)

        return tf.ragged.map_flat_values(gather_flat, indices)

    return tf.map_fn(
        gather_one,
        x[:, 0, :],
        fn_output_signature=tf.RaggedTensorSpec(
            shape=indices.shape,
            dtype=x.dtype,
            ragged_rank=indices.ragged_rank
        )
    )