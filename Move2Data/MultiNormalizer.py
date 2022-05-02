from pytorch_forecasting.data.encoders import TorchNormalizer, GroupNormalizer
from typing import Callable, Dict, Iterable, List, Tuple, Union
import pandas as pd
import numpy as np
import torch


class MultiNormalizer(TorchNormalizer):
    """
    Normalizer for multiple targets.

    This normalizers wraps multiple other normalizers.

    This is the debugged Normalizer, originally from pyTorch.
    """

    def __init__(self, normalizers: List[TorchNormalizer]):
        """
        Args:
            normalizers (List[TorchNormalizer]): list of normalizers to apply to targets
        """
        self.normalizers = normalizers

    def fit(self, y: Union[pd.DataFrame, np.ndarray, torch.Tensor], X: pd.DataFrame = None):
        """
        Fit transformer, i.e. determine center and scale of data

        Args:
            y (Union[pd.Series, np.ndarray, torch.Tensor]): input data

        Returns:
            MultiNormalizer: self
        """
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        for idx, normalizer in enumerate(self.normalizers):
            if isinstance(normalizer, GroupNormalizer):
                normalizer.fit(y[:, idx], X=X)
            else:
                normalizer.fit(y[:, idx])

        ### NEW ATTRIBUTE SET###
        self.fitted_ = True
        ######################
        return self

    def __getitem__(self, idx: int):
        """
        Return normalizer.

        Args:
            idx (int): metric index
        """
        return self.normalizers[idx]

    def __iter__(self):
        """
        Iter over normalizers.
        """
        return iter(self.normalizers)

    def __len__(self) -> int:
        """
        Number of normalizers.
        """
        return len(self.normalizers)

    def transform(
        self,
        y: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        X: pd.DataFrame = None,
        return_norm: bool = False,
        target_scale: List[torch.Tensor] = None,
    ) -> Union[List[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]], List[Union[np.ndarray, torch.Tensor]]]:
        """
        Scale input data.

        Args:
            y (Union[pd.DataFrame, np.ndarray, torch.Tensor]): data to scale
            X (pd.DataFrame): dataframe with ``groups`` columns. Only necessary if :py:class:`~GroupNormalizer`
                is among normalizers
            return_norm (bool, optional): If to return . Defaults to False.
            target_scale (List[torch.Tensor]): target scale to use instead of fitted center and scale

        Returns:
            Union[List[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]], List[Union[np.ndarray, torch.Tensor]]]:
                List of scaled data, if ``return_norm=True``, returns also scales as second element
        """
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy().transpose()

        res = []
        for idx, normalizer in enumerate(self.normalizers):
            if target_scale is not None:
                scale = target_scale[idx]
            else:
                scale = None
            if isinstance(normalizer, GroupNormalizer):
                r = normalizer.transform(y[idx], X=X, return_norm=return_norm, target_scale=scale)
            else:
                r = normalizer.transform(y[idx], return_norm=return_norm, target_scale=scale)
            res.append(r)

        if return_norm:
            return [r[0] for r in res], [r[1] for r in res]
        else:
            return res


    def __call__(self, data: Dict[str, Union[List[torch.Tensor], torch.Tensor]]) -> List[torch.Tensor]:
        """
        Inverse transformation but with network output as input.

        Args:
            data (Dict[str, Union[List[torch.Tensor], torch.Tensor]]): Dictionary with entries
                * prediction: list of data to de-scale
                * target_scale: list of center and scale of data

        Returns:
            List[torch.Tensor]: list of de-scaled data
        """
        denormalized = [
            normalizer(dict(prediction=data["prediction"][idx], target_scale=data["target_scale"][idx]))
            for idx, normalizer in enumerate(self.normalizers)
        ]
        return denormalized

    def get_parameters(self, *args, **kwargs) -> List[torch.Tensor]:
        """
        Returns parameters that were used for encoding.

        Returns:
            List[torch.Tensor]: First element is center of data and second is scale
        """
        return [normalizer.get_parameters(*args, **kwargs) for normalizer in self.normalizers]


    def __getattr__(self, name: str):
        """
        Return dynamically attributes.

        Return attributes if defined in this class. If not, create dynamically attributes based on
        attributes of underlying normalizers that are lists. Create functions if necessary.
        Arguments to functions are distributed to the functions if they are lists and their length
        matches the number of normalizers. Otherwise, they are directly passed to each callable of the
        normalizers.

        Args:
            name (str): name of attribute

        Returns:
            attributes of this class or list of attributes of underlying class
        """
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            attribute_exists = all([hasattr(norm, name) for norm in self.normalizers])
            if attribute_exists:
                # check if to return callable or not and return function if yes
                if callable(getattr(self.normalizers[0], name)):
                    n = len(self.normalizers)

                    def func(*args, **kwargs):
                        # if arg/kwarg is list and of length normalizers, then apply each part to a normalizer.
                        #  otherwise pass it directly to all normalizers
                        results = []
                        for idx, norm in enumerate(self.normalizers):
                            new_args = [
                                arg[idx]
                                if isinstance(arg, (list, tuple))
                                and not isinstance(arg, rnn.PackedSequence)
                                and len(arg) == n
                                else arg
                                for arg in args
                            ]
                            new_kwargs = {
                                key: val[idx]
                                if isinstance(val, list) and not isinstance(val, rnn.PackedSequence) and len(val) == n
                                else val
                                for key, val in kwargs.items()
                            }
                            results.append(getattr(norm, name)(*new_args, **new_kwargs))
                        return results

                    return func
                else:
                    # else return list of attributes
                    return [getattr(norm, name) for norm in self.normalizers]
            else:  # attribute does not exist for all normalizers
                raise e