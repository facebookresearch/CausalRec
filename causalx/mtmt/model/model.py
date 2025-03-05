import torch
from torch import nn
import pdb
from model.layer import CGC2D_Layer, MLP_Block, MMoE_Layer, get_output_activation, PositionalEncoding, Transformer_Block
from typing import List


class MLP(nn.Module):
    """
    Single outcome TARNet for test.
    """

    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, data) -> None:
        x = data["feature"]
        pred = torch.sigmoid(self.layers(x)).unsqueeze(1)
        return pred
    

class TARNet(nn.Module):
    """
    Multi-outcome multi-treatment TARNet
    """

    def __init__(self, input_dim, num_outcome, num_treatment, task="binary_classification") -> None:
        super().__init__()
        self.num_outcome = num_outcome
        self.num_treatment = num_treatment
        dim_hypo = 32  # Increased from 4
        
        # Shared representation network (only takes X, not T)
        self.hypothesis = nn.Sequential(
            nn.Linear(input_dim, 64),  # Remove T from input
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, dim_hypo),
        )
        
        # Each treatment has a head that predicts all outcomes
        self.treatment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hypo, 32),
                nn.ReLU(),
                nn.Linear(32, num_outcome),
            ) for _ in range(num_treatment)
        ])
        
        # Output activation configuration
        if isinstance(task, list):
            assert len(task) == num_outcome
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcome)]
            )

    def forward(self, data):
        x = data["feature"]
        
        # Get shared representation of X
        hypo = self.hypothesis(x)
        
        # Each treatment head predicts all outcomes
        treatment_preds = []
        for t in range(self.num_treatment):
            # Get predictions for all outcomes under this treatment
            outcomes = self.treatment_heads[t](hypo)  # [batch, num_outcomes]
            
            # Apply output activations for each outcome
            activated_outcomes = torch.stack([
                self.output_activation[o](outcomes[:, o])
                for o in range(self.num_outcome)
            ], dim=1)  # [batch, num_outcomes]
            
            treatment_preds.append(activated_outcomes)
            
        # Stack predictions from all treatments
        pred = torch.stack(treatment_preds, dim=2)  # [batch, num_outcomes, num_treatments]
        return pred
    

class CFRNet(nn.Module):
    """
    Multi-outcome multi-treatment TARNet
    """

    def __init__(self, input_dim, num_outcome, num_treatment, task="binary_classification") -> None:
        super().__init__()
        self.num_outcome = num_outcome
        self.num_treatment = num_treatment
        dim_hypo = 32  # Increased from 4
        
        # Shared representation network (only takes X, not T)
        self.hypothesis = nn.Sequential(
            nn.Linear(input_dim, 64),  # Remove T from input
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, dim_hypo),
        )
        
        # Each treatment has a head that predicts all outcomes
        self.treatment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hypo, 32),
                nn.ReLU(),
                nn.Linear(32, num_outcome),
            ) for _ in range(num_treatment)
        ])
        
        # Output activation configuration
        if isinstance(task, list):
            assert len(task) == num_outcome
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcome)]
            )

    def forward(self, data):
        x = data["feature"]
        
        # Get shared representation of X
        hypo = self.hypothesis(x)
        
        # Each treatment head predicts all outcomes
        treatment_preds = []
        for t in range(self.num_treatment):
            # Get predictions for all outcomes under this treatment
            outcomes = self.treatment_heads[t](hypo)  # [batch, num_outcomes]
            
            # Apply output activations for each outcome
            activated_outcomes = torch.stack([
                self.output_activation[o](outcomes[:, o])
                for o in range(self.num_outcome)
            ], dim=1)  # [batch, num_outcomes]
            
            treatment_preds.append(activated_outcomes)
            
        # Stack predictions from all treatments
        pred = torch.stack(treatment_preds, dim=2)  # [batch, num_outcomes, num_treatments]
        return pred, hypo


class Slearner_uplift(nn.Module):
    """
    Multi-outcome multi-treatment Slearner
    """
    
    def __init__(self, input_dim, num_outcome) -> None:
        super().__init__()
        self.num_outcome = num_outcome
        self.outcome_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ) for _ in range(num_outcome)
        ])
    
    def forward(self, data):
        x = data["feature"]
        pred_lst = [torch.sigmoid(head(x)) for head in self.outcome_heads]
        pred = torch.stack(pred_lst, dim=1)  # [batch x n_outcomes x 1]
        return pred

class Slearner(nn.Module):
    """
    Multi-outcome multi-treatment S-learner
    Treats treatment as another input feature and learns a single model
    """

    def __init__(self, input_dim, num_outcome, num_treatment, task="binary_classification") -> None:
        super().__init__()
        self.num_outcome = num_outcome
        self.num_treatment = num_treatment

        # Configure output activation based on task
        if isinstance(task, list):
            assert len(task) == num_outcome, "Length of task list must match num_outcome"
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcome)]
            )

        # Single network that takes both features and treatment as input
        # and outputs all outcomes at once
        self.network = nn.Sequential(
            nn.Linear(input_dim + num_treatment, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_outcome),
        )

    def forward(self, data):
        x = data["feature"]  # [batch, input_dim]
        batch_size = x.shape[0]
        
        # Generate predictions for all treatments
        treatment_preds = []
        for t in range(self.num_treatment):
            # Create one-hot treatment vector
            t_onehot = torch.zeros(batch_size, self.num_treatment, device=x.device)
            t_onehot[:, t] = 1
            
            # Concatenate features with treatment
            x_t = torch.cat([x, t_onehot], dim=1)  # [batch, input_dim + num_treatment]
            
            # Get predictions for all outcomes under this treatment
            pred = self.network(x_t)  # [batch, num_outcomes]
            
            # Apply output activations for each outcome without adding extra dimensions
            pred = torch.stack([
                self.output_activation[o](pred[:, o]) 
                for o in range(self.num_outcome)
            ], dim=1)  # [batch, num_outcomes]
            
            treatment_preds.append(pred)
        
        # Stack predictions from all treatments
        pred = torch.stack(treatment_preds, dim=2)  # [batch, num_outcomes, num_treatments]
        
        return pred

class Tlearner(nn.Module):
    """
    Multi-outcome multi-treatment T-learner
    Uses separate networks for each treatment condition
    """

    def __init__(self, input_dim, num_outcome, num_treatment, task="binary_classification") -> None:
        super().__init__()
        self.num_outcome = num_outcome
        self.num_treatment = num_treatment

        # Configure output activation based on task
        if isinstance(task, list):
            assert len(task) == num_outcome, "Length of task list must match num_outcome"
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcome)]
            )

        # Separate networks for each treatment, each predicting all outcomes
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_outcome),
            ) for _ in range(num_treatment)
        ])

    def forward(self, data):
        x = data["feature"]
        
        # Get predictions for each treatment
        treatment_preds = []
        for t in range(self.num_treatment):
            # Get predictions for all outcomes under this treatment
            pred = self.networks[t](x)  # [batch, num_outcomes]
            
            # Apply output activations for each outcome
            pred = torch.stack([
                self.output_activation[o](pred[:, o]) 
                for o in range(self.num_outcome)
            ], dim=1)  # [batch, num_outcomes]
            
            treatment_preds.append(pred)
            
        # Stack predictions from all treatments
        pred = torch.stack(treatment_preds, dim=2)  # [batch, num_outcomes, num_treatments]
        
        return pred



class S_MMoE(nn.Module):
    def __init__(self,
                 num_outcome=1,
                 num_experts=4,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 input_dim=1,
                 task="binary_classification"
                ):
        super().__init__()
        self.num_outcome = num_outcome
        self.output_activation = get_output_activation(task)
        if isinstance(task, list):
            assert len(task) == num_outcome, "the number of outcomes must equal the length of \"task\""
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcome)]
            )
        self.mmoe_layer = MMoE_Layer(num_experts=num_experts,
                                     num_tasks=self.num_outcome,
                                     input_dim=input_dim,
                                     expert_hidden_units=expert_hidden_units,
                                     gate_hidden_units=gate_hidden_units,
                                     hidden_activations=hidden_activations,
                                     net_dropout=net_dropout,
                                     batch_norm=batch_norm)
        self.tower = nn.ModuleList([MLP_Block(input_dim=expert_hidden_units[-1],
                                              output_dim=1,
                                              hidden_units=tower_hidden_units,
                                              hidden_activations=hidden_activations,
                                              output_activation=None,
                                              dropout_rates=net_dropout,
                                              batch_norm=batch_norm)
                                    for _ in range(num_outcome)])

    def forward(self, data):
        x = data["feature"]
        expert_output = self.mmoe_layer(x)
        tower_output = [self.tower[i](expert_output[i]) for i in range(self.num_outcome)]
        pred_lst = [self.output_activation[i](tower_output[i]) for i in range(self.num_outcome)]
        pred = torch.stack(pred_lst, dim=1)  # [batch x n_outcomes x 1]
        return pred

# num_towers = num_outcomes x num_treatments
class Vanilla_MMoE(nn.Module):
    def __init__(self,
                 num_outcomes=1,
                 num_treatments=2,
                 num_experts=4,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 input_dim=1,
                 task="binary_classification"
                ):
        super().__init__()
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.output_activation = get_output_activation(task)
        if isinstance(task, list):
            assert len(task) == num_outcomes*num_treatments, "the number of outcomes x treatments must equal the length of \"task\""
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList([
                nn.ModuleList([
                    get_output_activation(task) 
                    for _ in range(num_treatments)
                ]) 
                for _ in range(num_outcomes)
            ])
    
        self.mmoe_layer = MMoE_Layer(num_experts=num_experts,
                                     num_tasks=self.num_outcomes * self.num_treatments,
                                     input_dim=input_dim,
                                     expert_hidden_units=expert_hidden_units,
                                     gate_hidden_units=gate_hidden_units,
                                     hidden_activations=hidden_activations,
                                     net_dropout=net_dropout,
                                     batch_norm=batch_norm)
        self.tower = nn.ModuleList([
                        nn.ModuleList([
                            MLP_Block(
                                input_dim=expert_hidden_units[-1],
                                output_dim=1,
                                hidden_units=tower_hidden_units,
                                hidden_activations=hidden_activations,
                                output_activation=None,
                                dropout_rates=net_dropout,
                                batch_norm=batch_norm
                            )
                            for _ in range(num_treatments)
                        ])
                        for _ in range(num_outcomes)
                    ])

    def forward(self, data):
        x = data["feature"]
        expert_output = self.mmoe_layer(x)
        expert_output = [expert_output[i * self.num_treatments:(i + 1) * self.num_treatments] for i in range(self.num_outcomes)]
        tower_output = [[self.tower[i][j](expert_output[i][j]) for j in range(self.num_treatments)] for i in range(self.num_outcomes)]
        pred_lst = [[self.output_activation[i][j](tower_output[i][j]) for j in range(self.num_treatments)] for i in range(self.num_outcomes)]
        pred = torch.stack([torch.cat(pred_lst[i], dim=1) for i in range(self.num_outcomes)], dim=1) # [batch x n_outcomes x num_treatments]
        return pred
    

class PLE2D(nn.Module):
    def __init__(self,
                 num_layers=1,
                 expert_type="mlp",
                 dcnv2_model_structure="stacked",
                 dcnv2_num_cross_layers=2,
                 num_shared_experts=1,
                 num_outcome_shared_experts=1,
                 num_treatment_shared_experts=1,
                 num_specific_experts=1,
                 num_outcomes=1,
                 num_treatments=2,
                 input_hidden_units=[16],
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 input_dim=1,
                 task="binary_classification"
                ):
        super().__init__()
        self.num_layers = num_layers
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.output_activation = get_output_activation(task)
        if isinstance(task, list):
            assert len(task) == num_outcomes*num_treatments, "the number of outcomes x treatments must equal the length of \"task\""
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList([
                nn.ModuleList([
                    get_output_activation(task) 
                    for _ in range(num_treatments)
                ]) 
                for _ in range(num_outcomes)
            ])
        self.input_layers = MLP_Block(
                                input_dim=input_dim,
                                hidden_units=input_hidden_units,
                                hidden_activations=hidden_activations,
                                output_activation=None,
                                dropout_rates=net_dropout,
                                batch_norm=batch_norm
                            )
        self.cgc_layers = nn.ModuleList(
            [CGC2D_Layer(
                expert_type,
                num_shared_experts, 
                num_outcome_shared_experts, 
                num_treatment_shared_experts, 
                num_specific_experts, 
                num_outcomes, 
                num_treatments, 
                input_hidden_units[-1] if i==0 else expert_hidden_units[-1],
                expert_hidden_units, 
                gate_hidden_units, 
                hidden_activations, 
                net_dropout, 
                batch_norm,
                dcnv2_model_structure,
                dcnv2_num_cross_layers
            ) for i in range(self.num_layers)]
        )
        self.tower = nn.ModuleList([
                        nn.ModuleList([
                            MLP_Block(
                                input_dim=expert_hidden_units[-1],
                                output_dim=1,
                                hidden_units=tower_hidden_units,
                                hidden_activations=hidden_activations,
                                output_activation=None,
                                dropout_rates=net_dropout,
                                batch_norm=batch_norm
                            )
                            for _ in range(num_treatments)
                        ])
                        for _ in range(num_outcomes)
                    ])

    def forward(self, data):
        x = data["feature"]
        x = self.input_layers(x)
        cgc_inputs = [x for _ in range(self.num_outcomes*self.num_treatments+self.num_outcomes+self.num_treatments+1)]
        for i in range(self.num_layers):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs
        cgc_outputs = [cgc_outputs[i * self.num_treatments:(i + 1) * self.num_treatments] for i in range(self.num_outcomes)]
        tower_output = [[self.tower[i][j](cgc_outputs[i][j]) for j in range(self.num_treatments)] for i in range(self.num_outcomes)]
        pred_lst = [[self.output_activation[i][j](tower_output[i][j]) for j in range(self.num_treatments)] for i in range(self.num_outcomes)]
        pred = torch.stack([torch.cat(pred_lst[i], dim=1) for i in range(self.num_outcomes)], dim=1) # [batch x n_outcomes x num_treatments]
        return pred
    
    def get_gate_matrices(self, data):
        x = data["feature"]
        x = self.input_layers(x)
        gate_matrices = []
        cgc_inputs = [x for _ in range(self.num_outcomes*self.num_treatments+self.num_outcomes+self.num_treatments+1)]
        for i in range(self.num_layers):
            cgc_outputs, gate_matrix = self.cgc_layers[i].get_gate_matrix(cgc_inputs)
            cgc_inputs = cgc_outputs
            gate_matrices.append(gate_matrix)
        return gate_matrices

    
class MultiTaskMoE(nn.Module):
    def __init__(self,
                 input_dim=1,
                 num_experts=5,
                 expert_hidden_units=[64, 32],
                 num_outcomes=10,
                 num_treatments=3,
                 gate_hidden_units=[32, 16],
                 hidden_activations="ELU",
                 net_dropout=0,
                 batch_norm=False,
                 task="binary_classification"):
        super(MultiTaskMoE, self).__init__()
        self.num_experts = num_experts
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments

        # Determine output activation per outcome
        if isinstance(task, list):
            assert len(task) == num_outcomes, "Length of task list must match num_outcomes."
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcomes)]
            )

        # Shared experts
        self.experts = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                output_dim=expert_hidden_units[-1],
                hidden_units=expert_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(num_experts)
        ])

        # Outcome-specific gating networks
        self.gates = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                output_dim=num_experts,
                hidden_units=gate_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=nn.Softmax(dim=1),
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(num_outcomes)
        ])

        # Treatment towers per expert
        # Each expert → num_treatments towers
        self.treatment_towers = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(
                    input_dim=expert_hidden_units[-1],
                    output_dim=1,
                    hidden_units=[expert_hidden_units[-1]//2, expert_hidden_units[-1]//4],
                    hidden_activations=hidden_activations,
                    output_activation=None,
                    dropout_rates=net_dropout,
                    batch_norm=batch_norm
                ) for _ in range(num_treatments)
            ]) for _ in range(num_experts)
        ])

        # Outcome-specific towers
        # After mixing experts for each outcome, we get [batch, num_treatments].
        # We can define a small MLP for each outcome that processes these combined predictions.
        # Input: num_treatments, Output: num_treatments (we keep dimension consistent).
        # You can adjust the hidden units as desired.
        self.outcome_towers = nn.ModuleList([
            MLP_Block(
                input_dim=num_treatments,
                output_dim=num_treatments,
                hidden_units=[num_treatments*2, num_treatments],
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(num_outcomes)
        ])

    def forward(self, data):
        x = data["feature"]  # [batch, input_dim]

        # Compute expert outputs: [batch, hidden_dim, num_experts]
        expert_reps = torch.stack([expert(x) for expert in self.experts], dim=2)

        # Compute treatment-specific outputs for each expert
        # [batch, num_experts, num_treatments]
        expert_treatment_outs = []
        for i in range(self.num_experts):
            rep_i = expert_reps[:, :, i]  # [batch, hidden_dim]
            treatment_preds = []
            for t in range(self.num_treatments):
                out_t = self.treatment_towers[i][t](rep_i)  # [batch, 1]
                treatment_preds.append(out_t.squeeze(-1))   # [batch]
            treatment_preds = torch.stack(treatment_preds, dim=1)  # [batch, num_treatments]
            expert_treatment_outs.append(treatment_preds)

        # [batch, num_experts, num_treatments]
        expert_treatment_outs = torch.stack(expert_treatment_outs, dim=1)

        outcome_outputs = []
        for o in range(self.num_outcomes):
            # Get gating weights for this outcome
            gate_weights = self.gates[o](x)  # [batch, num_experts]

            # Weighted sum over experts for each treatment
            # gate_weights: [batch, M], expand to [batch, M, 1]
            weighted_treatment_outs = (expert_treatment_outs * gate_weights.unsqueeze(-1)).sum(dim=1)
            # weighted_treatment_outs: [batch, num_treatments]

            # Pass through outcome-specific tower
            outcome_rep = self.outcome_towers[o](weighted_treatment_outs)  # [batch, num_treatments]

            # Apply final activation function
            activated_outs = self.output_activation[o](outcome_rep)  # [batch, num_treatments]

            outcome_outputs.append(activated_outs)

        # [batch, num_outcomes, num_treatments]
        pred = torch.stack(outcome_outputs, dim=1)

        return pred


class MultiTaskMoETransformer(nn.Module):
    def __init__(self,
                 input_dim=1,
                 num_experts=5,
                 expert_hidden_units=[64, 32],
                 num_outcomes=10,
                 num_treatments=3,
                 gate_hidden_units=[32, 16],
                 hidden_activations="ELU",
                 net_dropout=0,
                 batch_norm=False,
                 task="binary_classification",
                 gating_output_activation="softmax",  # New parameter for gating activation
                 top_k=None):  # Only used if gating_output_activation='top_k'
        super(MultiTaskMoETransformer, self).__init__()
        self.num_experts = num_experts
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments

        # Determine output activation per outcome
        if isinstance(task, list):
            assert len(task) == num_outcomes, "Length of task list must match num_outcomes."
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcomes)]
            )

        # Experts as Transformer blocks (no output activation for experts, just a representation)
        d_model = expert_hidden_units[-1]
        self.experts = nn.ModuleList([
            Transformer_Block(
                input_dim=input_dim,
                d_model=d_model,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=net_dropout,
                output_dim=None,             # Experts produce a hidden representation
                output_activation=None       # No activation here
            ) for _ in range(num_experts)
        ])

        # Gating networks as Transformer blocks with output_dim = num_experts and optional activation
        # If gating_output_activation='softmax', then softmax is applied to produce a probability distribution.
        # If gating_output_activation='top_k', then top-k gating is used (must specify top_k).
        self.gates = nn.ModuleList([
            Transformer_Block(
                input_dim=input_dim,
                d_model=num_experts,
                nhead=2,
                num_layers=1,
                dim_feedforward=64,
                dropout=net_dropout,
                output_dim=num_experts,                    # Gating must output a vector of expert scores
                output_activation=gating_output_activation, # e.g., 'softmax' or 'top_k'
                top_k=top_k
            ) for _ in range(num_outcomes)
        ])

        # Treatment towers (MLPs) remain the same
        self.treatment_towers = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(
                    input_dim=expert_hidden_units[-1],
                    output_dim=1,
                    hidden_units=[expert_hidden_units[-1]//2, expert_hidden_units[-1]//4],
                    hidden_activations=hidden_activations,
                    output_activation=None,
                    dropout_rates=net_dropout,
                    batch_norm=batch_norm
                ) for _ in range(num_treatments)
            ]) for _ in range(num_experts)
        ])

        # Outcome-specific towers (MLPs) remain optional and unchanged
        self.outcome_towers = nn.ModuleList([
            MLP_Block(
                input_dim=num_treatments,
                output_dim=num_treatments,
                hidden_units=[num_treatments*2, num_treatments],
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(num_outcomes)
        ])

    def forward(self, data):
        x = data["feature"]  # [batch, input_dim]

        # Compute expert representations: [batch, hidden_dim, num_experts]
        expert_outputs = [expert(x) for expert in self.experts]
        expert_reps = torch.stack(expert_outputs, dim=2)  # [batch, hidden_dim, num_experts]

        # Compute treatment-specific predictions for each expert
        expert_treatment_outs = []
        for i in range(self.num_experts):
            rep_i = expert_reps[:, :, i]  # [batch, hidden_dim]
            treatment_preds = []
            for t in range(self.num_treatments):
                out_t = self.treatment_towers[i][t](rep_i)  # [batch, 1]
                treatment_preds.append(out_t.squeeze(-1))   # [batch]
            treatment_preds = torch.stack(treatment_preds, dim=1)  # [batch, num_treatments]
            expert_treatment_outs.append(treatment_preds)

        # [batch, num_experts, num_treatments]
        expert_treatment_outs = torch.stack(expert_treatment_outs, dim=1)

        # Compute outcome-specific predictions
        outcome_outputs = []
        for o in range(self.num_outcomes):
            # Gating weights from Transformer-based gating networks
            gate_weights = self.gates[o](x)  # [batch, num_experts]
            # If output_activation='softmax', gate_weights are already probabilities.
            # If output_activation='top_k', gate_weights are top-k normalized.
            # If None, no activation applied (then you'd need to handle normalization).

            # Weighted sum of experts for each treatment
            weighted_treatment_outs = (expert_treatment_outs * gate_weights.unsqueeze(-1)).sum(dim=1)
            # weighted_treatment_outs: [batch, num_treatments]

            # Pass through outcome-specific tower
            outcome_rep = self.outcome_towers[o](weighted_treatment_outs)  # [batch, num_treatments]

            # Apply final activation function (e.g., sigmoid for binary_classification)
            activated_outs = self.output_activation[o](outcome_rep)  # [batch, num_treatments]

            outcome_outputs.append(activated_outs)

        # [batch, num_outcomes, num_treatments]
        pred = torch.stack(outcome_outputs, dim=1)
        return pred


# Gradient Reversal Layer
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.lambda_), None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class MultiTaskMoEDANN(nn.Module):
    def __init__(self,
                 input_dim=1,
                 num_experts=5,
                 expert_hidden_units=[64, 32],
                 num_outcomes=10,
                 num_treatments=3,
                 gate_hidden_units=[32, 16],
                 hidden_activations="ELU",
                 net_dropout=0,
                 batch_norm=False,
                 task="binary_classification",
                 domain_lambda=1.0):
        super(MultiTaskMoEDANN, self).__init__()
        self.num_experts = num_experts
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.domain_lambda = domain_lambda  # weighting for domain loss

        # Output activation
        if isinstance(task, list):
            assert len(task) == num_outcomes, "Length of task list must match num_outcomes."
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcomes)]
            )

        # Shared experts
        self.experts = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                output_dim=expert_hidden_units[-1],
                hidden_units=expert_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(num_experts)
        ])

        # Outcome-specific gating networks
        self.gates = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                output_dim=num_experts,
                hidden_units=gate_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=nn.Softmax(dim=1),
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(num_outcomes)
        ])

        # Treatment towers per expert
        self.treatment_towers = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(
                    input_dim=expert_hidden_units[-1],
                    output_dim=1,
                    hidden_units=[expert_hidden_units[-1]//2, expert_hidden_units[-1]//4],
                    hidden_activations=hidden_activations,
                    output_activation=None,
                    dropout_rates=net_dropout,
                    batch_norm=batch_norm
                ) for _ in range(num_treatments)
            ]) for _ in range(num_experts)
        ])

        # Outcome-specific towers
        self.outcome_towers = nn.ModuleList([
            MLP_Block(
                input_dim=num_treatments,
                output_dim=num_treatments,
                hidden_units=[num_treatments*2, num_treatments],
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(num_outcomes)
        ])

        # Domain discriminator (DANN part)
        # Takes the average expert representation and tries to predict the treatment domain
        self.domain_discriminator = MLP_Block(
            input_dim=expert_hidden_units[-1],
            output_dim=num_treatments,
            hidden_units=[64, 32],  # customize as needed
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        )

    def forward(self, data):
        x = data["feature"]  # [batch, input_dim]

        # Compute expert outputs: [batch, hidden_dim, num_experts]
        expert_reps = torch.stack([expert(x) for expert in self.experts], dim=2)  # [B, H, M]

        # For domain adversarial: create a domain prediction
        # We'll take the mean across experts to get a shared representation
        expert_reps_mean = expert_reps.mean(dim=2)  # [B, H]

        # Apply GRL before domain discriminator
        reversed_feats = grad_reverse(expert_reps_mean, self.domain_lambda)
        domain_preds = self.domain_discriminator(reversed_feats)  # [B, num_treatments]

        # Compute treatment-specific outputs per expert
        expert_treatment_outs = []
        for i in range(self.num_experts):
            rep_i = expert_reps[:, :, i]  # [batch, hidden_dim]
            treatment_preds = []
            for t in range(self.num_treatments):
                out_t = self.treatment_towers[i][t](rep_i)  # [batch, 1]
                treatment_preds.append(out_t.squeeze(-1))   # [batch]
            treatment_preds = torch.stack(treatment_preds, dim=1)  # [batch, num_treatments]
            expert_treatment_outs.append(treatment_preds)

        # [batch, num_experts, num_treatments]
        expert_treatment_outs = torch.stack(expert_treatment_outs, dim=1)

        outcome_outputs = []
        for o in range(self.num_outcomes):
            # Get gating weights for this outcome
            gate_weights = self.gates[o](x)  # [batch, num_experts]

            # Weighted sum over experts for each treatment
            weighted_treatment_outs = (expert_treatment_outs * gate_weights.unsqueeze(-1)).sum(dim=1)
            # weighted_treatment_outs: [batch, num_treatments]

            # Pass through outcome-specific tower
            outcome_rep = self.outcome_towers[o](weighted_treatment_outs)  # [batch, num_treatments]

            # Apply final activation function
            activated_outs = self.output_activation[o](outcome_rep)  # [batch, num_treatments]

            outcome_outputs.append(activated_outs)

        # [batch, num_outcomes, num_treatments]
        pred = torch.stack(outcome_outputs, dim=1)

        # Return predictions and domain predictions
        return pred, domain_preds

class PLE2D_DA(PLE2D):
    def __init__(self,
                 num_layers=1,
                 expert_type="mlp",
                 dcnv2_model_structure="stacked",
                 dcnv2_num_cross_layers=2,
                 num_shared_experts=1,
                 num_outcome_shared_experts=1,
                 num_treatment_shared_experts=1,
                 num_specific_experts=1,
                 num_outcomes=1,
                 num_treatments=2,
                 input_hidden_units=[16],
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 input_dim=1,
                 task="binary_classification"):
        super().__init__(
            num_layers=num_layers,
            expert_type=expert_type,
            dcnv2_model_structure=dcnv2_model_structure,
            dcnv2_num_cross_layers=dcnv2_num_cross_layers,
            num_shared_experts=num_shared_experts,
            num_outcome_shared_experts=num_outcome_shared_experts,
            num_treatment_shared_experts=num_treatment_shared_experts,
            num_specific_experts=num_specific_experts,
            num_outcomes=num_outcomes,
            num_treatments=num_treatments,
            input_hidden_units=input_hidden_units,
            expert_hidden_units=expert_hidden_units,
            gate_hidden_units=gate_hidden_units,
            tower_hidden_units=tower_hidden_units,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            input_dim=input_dim,
            task=task
        )

    def forward(self, data):
        x = data["feature"]
        x = self.input_layers(x)
        cgc_inputs = [x for _ in range(self.num_outcomes*self.num_treatments+self.num_outcomes+self.num_treatments+1)]
        
        # Process through CGC layers
        for i in range(self.num_layers - 1):
            cgc_outputs = self.cgc_layers[i](cgc_inputs)  # Don't need treatment_shared_outputs except last layer
            cgc_inputs = cgc_outputs

        # Only get treatment_shared_outputs from final layer
        cgc_outputs, treatment_shared_outputs = self.cgc_layers[-1](cgc_inputs, require_treatment_shared=True)

        # Process final outputs as in original PLE2D
        cgc_outputs = [cgc_outputs[i * self.num_treatments:(i + 1) * self.num_treatments] for i in range(self.num_outcomes)]
        tower_output = [[self.tower[i][j](cgc_outputs[i][j]) for j in range(self.num_treatments)] for i in range(self.num_outcomes)]
        pred_lst = [[self.output_activation[i][j](tower_output[i][j]) for j in range(self.num_treatments)] for i in range(self.num_outcomes)]
        pred = torch.stack([torch.cat(pred_lst[i], dim=1) for i in range(self.num_outcomes)], dim=1)
        #print(f"treatment_shared_outputs shape: {treatment_shared_outputs.shape}")
        return pred, treatment_shared_outputs  # treatment_shared_outputs is [num_experts, batch, hidden_dim]ç

class DragonNet(nn.Module):
    """
    Multi-outcome multi-treatment DragonNet based on TARNet architecture.
    Jointly learns propensity scores and potential outcomes with a shared representation.
    """

    def __init__(self, input_dim, num_outcome, num_treatment, task="binary_classification") -> None:
        super().__init__()
        self.num_outcome = num_outcome
        self.num_treatment = num_treatment
        dim_hypo = 32
        
        # Shared representation network (similar to TARNet)
        self.hypothesis = nn.Sequential(
            nn.Linear(input_dim, 64),  # Remove T from input
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, dim_hypo),
        )
        
        # Propensity network (treatment prediction)
        self.propensity_net = nn.Sequential(
            nn.Linear(dim_hypo, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_treatment),
            nn.Softmax(dim=1)  # Outputs probabilities for each treatment
        )
        
        # Treatment-specific outcome heads (like TARNet)
        self.treatment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hypo, 32),
                nn.ReLU(),
                nn.Linear(32, num_outcome),
            ) for _ in range(num_treatment)
        ])
        
        # Output activation configuration
        if isinstance(task, list):
            assert len(task) == num_outcome
            self.output_activation = nn.ModuleList([get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [get_output_activation(task) for _ in range(num_outcome)]
            )

    def forward(self, data):
        x = data["feature"]
        
        # Get shared representation
        hypo = self.hypothesis(x)
        
        # Get propensity scores
        propensity_scores = self.propensity_net(hypo)
        
        # Each treatment head predicts all outcomes
        treatment_preds = []
        for t in range(self.num_treatment):
            # Get predictions for all outcomes under this treatment
            outcomes = self.treatment_heads[t](hypo)  # [batch, num_outcomes]
            
            # Apply output activations for each outcome
            activated_outcomes = torch.stack([
                self.output_activation[o](outcomes[:, o])
                for o in range(self.num_outcome)
            ], dim=1)  # [batch, num_outcomes]
            
            treatment_preds.append(activated_outcomes)
            
        # Stack predictions from all treatments
        pred = torch.stack(treatment_preds, dim=2)  # [batch, num_outcomes, num_treatments]
        
        return pred, propensity_scores, hypo
