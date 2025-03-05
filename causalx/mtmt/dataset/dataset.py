import torch
from torch.utils.data import Dataset

class UpliftDataset(Dataset):
    def __init__(
        self,
        data_df,
        config,
        device,
    ):
        self.config = config
        self.feature = data_df[config["feature_cols"]].values
        self.trt = data_df[config["treatment_col"]].values
        self.outcome = data_df[config["outcome_cols"]].values
        self.feature_processed = False
        self.outcome_processed = False
        self.device = device
    
    def __len__(self):
        return len(self.trt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        if self.trt is not None:
            sample["trt"] = torch.tensor(self.trt[idx], dtype=torch.int64).to(self.device)
        if self.feature is not None:
            sample["feature"] = torch.tensor(self.feature[idx], dtype=torch.float).to(self.device)
        if self.outcome is not None:
            sample["outcome"] = torch.tensor(self.outcome[idx], dtype=torch.float).to(self.device)
        return sample
    
    def feature_fit(self, processor):
        processor.fit(self.feature)
    
    def feature_transform(self, processor):
        self.feature = processor.transform(self.feature)
        self.feature_processed = True
        
    def outcome_fit(self, processor):
        processor.fit(self.outcome)
    
    def outcome_transform(self, processor):
        self.outcome = processor.transform(self.outcome)
        self.outcome_processed = True
        
    def get_feature(self):
        return self.feature
    
    def get_outcome(self):
        return self.outcome
    
    def get_trt(self):
        return self.trt
    
    def assign_trt(self, val):
        # assume the trt is the last feature
        self.feature[:, -1] = val
        