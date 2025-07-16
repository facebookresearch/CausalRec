# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Union, List, Optional

class UpliftDataset(Dataset):
    """Dataset class for uplift modeling.
    
    Args:
        data_df: Pandas DataFrame containing features, treatment, and outcome data
        config: Dictionary containing column configurations with keys:
            - feature_cols: List of feature column names
            - treatment_col: Treatment column name
            - outcome_cols: List of outcome column names
        device: torch device ('cpu' or 'cuda')
    """
    def __init__(
        self,
        data_df,
        config: Dict[str, Union[str, List[str]]],
        device: str,
    ):
        if not all(key in config for key in ["feature_cols", "treatment_col", "outcome_cols"]):
            raise ValueError("Config must contain 'feature_cols', 'treatment_col', and 'outcome_cols'")

        self.config = config
        self.device = device
        self.feature_processed = False
        self.outcome_processed = False

        # Convert to numpy arrays and store
        self._feature = data_df[config["feature_cols"]].values.astype(np.float32)
        self._trt = data_df[config["treatment_col"]].values.astype(np.int64)
        self._outcome = data_df[config["outcome_cols"]].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self._trt)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {
            "trt": torch.tensor(self._trt[idx], dtype=torch.int64, device=self.device),
            "feature": torch.tensor(self._feature[idx], dtype=torch.float32, device=self.device),
            "outcome": torch.tensor(self._outcome[idx], dtype=torch.float32, device=self.device)
        }

    def feature_fit(self, processor) -> None:
        """Fit the feature by processor."""
        processor.fit(self.feature)
    
    def feature_transform(self, processor) -> None:
        """Transform features using the fitted processor."""
        self.feature = processor.transform(self.feature)
        self.feature_processed = True
        
    def outcome_fit(self, processor) -> None:
        """Fit the outcome processor."""
        processor.fit(self.outcome)
    
    def outcome_transform(self, processor) -> None:
        """Transform outcomes using the fitted processor."""
        self.outcome = processor.transform(self.outcome)
        self.outcome_processed = True

    @property
    def feature(self) -> np.ndarray:
        return self._feature

    @feature.setter
    def feature(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Feature must be a numpy array")
        if value.dtype != np.float32:
            value = value.astype(np.float32)
        self._feature = value

    @property
    def outcome(self) -> np.ndarray:
        return self._outcome

    @outcome.setter
    def outcome(self, value: np.ndarray) -> None:
        self._outcome = value

    @property
    def trt(self) -> np.ndarray:
        return self._trt

    @trt.setter
    def trt(self, value: np.ndarray) -> None:
        self._trt = value
        
    def assign_trt(self, val: Union[float, np.ndarray]) -> None:
        """Assign treatment value(s) to the last feature column.
        
        Args:
            val: Value or array of values to assign as treatment
        """
        self.feature[:, -1] = val

    def to_device(self, device: str) -> None:
        """Move dataset to specified device (CPU or GPU).
        
        Args:
            device: Target device ('cpu' or 'cuda')
        """
        self.device = device


def build_dataset(data_df, config, is_train, device, processors=None):
    # Initialize processors as empty dict if None
    if processors is None:
        processors = {"feature": None, "outcome": None}
        
    dataset = UpliftDataset(data_df, config, device)
    
    # Handle feature processing
    if is_train:
        # Create feature processor based on config
        processor_type = config.get("processor_type", "minmax")
        if processor_type == "minmax":
            processors["feature"] = MinMaxScaler(feature_range=(0, 1))
        elif processor_type == "standard":
            processors["feature"] = StandardScaler()
        else:
            raise ValueError("Invalid processor type specified in the config. Use 'minmax' or 'standard'.")
        dataset.feature_fit(processors["feature"])
    elif processors["feature"] is None:
        raise ValueError("Feature processor must be provided for test dataset.")
    
    # Apply feature transformation
    dataset.feature_transform(processors["feature"])
    
    # Handle outcome processing if enabled
    use_outcome_proc = config.get("use_outcome_proc", False)
    if use_outcome_proc:
        if is_train:
            # Create outcome processor based on config
            outcome_processor_type = config.get("outcome_processor_type", "minmax")
            if outcome_processor_type == "minmax":
                processors["outcome"] = MinMaxScaler(feature_range=(0, 1))
            elif outcome_processor_type == "standard":
                processors["outcome"] = StandardScaler()
            else:
                raise ValueError("Invalid outcome processor type specified in the config. Use 'minmax' or 'standard'.")
            dataset.outcome_fit(processors["outcome"])
        elif processors["outcome"] is None:
            raise ValueError("Outcome processor must be provided for test dataset.")
        
        # Apply outcome transformation
        dataset.outcome_transform(processors["outcome"])
        
    return dataset, processors
        
