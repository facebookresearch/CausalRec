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
from torch import nn
import math
import torch.nn.functional as F

class MLP_Block(nn.Module):
    """
    Adopted from https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/pytorch/layers/blocks/mlp_block.py
    """
    def __init__(self, 
                 input_dim, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None, 
                 dropout_rates=0.0,
                 batch_norm=False, 
                 bn_only_once=False, # Set True for inference speed up
                 use_bias=True):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        if batch_norm and bn_only_once:
            dense_layers.append(nn.BatchNorm1d(input_dim))
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm and not bn_only_once:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.mlp(inputs)
    

class MMoE_Layer(nn.Module):
    """
    Adopted from https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/multitask/MMoE/src/MMoE.py
    """
    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        super(MMoE_Layer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                                hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_experts)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                             hidden_units=gate_hidden_units,
                                             output_dim=num_experts,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for _ in range(self.num_tasks)])
        self.gate_activation = get_activation('softmax')

    def forward(self, x):
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)],
                                     dim=1)  # (?, num_experts, dim)
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](x)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (?, num_experts)
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
        return mmoe_output


class CrossNetV2(nn.Module):
    """
    Adopted from https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/pytorch/layers/interactions/cross_net.py
    """
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i
        

class DCNv2(nn.Module):
    """
    Adopted from https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/DCNv2/src/DCNv2.py
    """
    def __init__(self,
                 input_dim,
                 model_structure="stacked",
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 **kwargs):
        super(DCNv2, self).__init__(**kwargs)
        self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"], \
               "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(input_dim=input_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=stacked_dnn_hidden_units,
                                         hidden_activations=dnn_activations,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            final_dim = stacked_dnn_hidden_units[-1]
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure == "crossnet_only": # only CrossNet
            final_dim = input_dim
        self.fc = nn.Linear(final_dim, 1)

    def forward(self, inputs):
        cross_out = self.crossnet(inputs)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(inputs)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(inputs)], dim=-1)
        return final_out


class CGC2D_Layer(nn.Module):
    """
    Modified from https://github.com/reczoo/FuxiCTR/blob/562061c94ca218fea41b5dc4b15230475bd99cb0/model_zoo/multitask/PLE/src/PLE.py
    """
    def __init__(self, expert_type, num_shared_experts, num_outcome_shared_experts, num_treatment_shared_experts, num_specific_experts, 
                 num_outcomes, num_treatments, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations, net_dropout, batch_norm, dcnv2_model_structure, dcnv2_num_cross_layers):
        super(CGC2D_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_outcome_shared_experts = num_outcome_shared_experts
        self.num_treatment_shared_experts = num_treatment_shared_experts
        self.num_specific_experts = num_specific_experts 
        self.num_tasks = num_outcomes * num_treatments
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        if expert_type == "mlp":
            self.shared_experts = nn.ModuleList(
                [MLP_Block(input_dim=input_dim,
                hidden_units=expert_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm) for _ in range(self.num_shared_experts)]
            )
            self.outcome_shared_experts = nn.ModuleList(
                [nn.ModuleList([MLP_Block(input_dim=input_dim,
                hidden_units=expert_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm) for _ in range(self.num_outcome_shared_experts)]) for _ in range(num_treatments)]
            )
            self.treatment_shared_experts = nn.ModuleList(
                [nn.ModuleList([MLP_Block(input_dim=input_dim,
                hidden_units=expert_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm) for _ in range(self.num_treatment_shared_experts)]) for _ in range(num_outcomes)]
            )
            if num_specific_experts > 0:
                self.specific_experts = nn.ModuleList(
                    [nn.ModuleList([MLP_Block(input_dim=input_dim,
                    hidden_units=expert_hidden_units,
                    hidden_activations=hidden_activations,
                    output_activation=None,
                    dropout_rates=net_dropout,
                    batch_norm=batch_norm) for _ in range(self.num_specific_experts)]) for _ in range(self.num_tasks)]
                )
        elif expert_type == "dcnv2":
            self.shared_experts = nn.ModuleList([DCNv2(
                input_dim=input_dim,
                model_structure=dcnv2_model_structure,
                stacked_dnn_hidden_units=expert_hidden_units,
                parallel_dnn_hidden_units=expert_hidden_units,
                dnn_activations=hidden_activations,
                num_cross_layers=dcnv2_num_cross_layers,
                net_dropout=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(self.num_shared_experts)])
            self.outcome_shared_experts = nn.ModuleList([nn.ModuleList([DCNv2(
                input_dim=input_dim,
                model_structure=dcnv2_model_structure,
                stacked_dnn_hidden_units=expert_hidden_units,
                parallel_dnn_hidden_units=expert_hidden_units,
                dnn_activations=hidden_activations,
                num_cross_layers=dcnv2_num_cross_layers,
                net_dropout=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(self.num_outcome_shared_experts)]) for _ in range(self.num_treatments)])
            self.treatment_shared_experts = nn.ModuleList([nn.ModuleList([DCNv2(
                input_dim=input_dim,
                model_structure=dcnv2_model_structure,
                stacked_dnn_hidden_units=expert_hidden_units,
                parallel_dnn_hidden_units=expert_hidden_units,
                dnn_activations=hidden_activations,
                num_cross_layers=dcnv2_num_cross_layers,
                net_dropout=net_dropout,
                batch_norm=batch_norm
            ) for _ in range(self.num_treatment_shared_experts)]) for _ in range(self.num_outcomes)])
            if num_specific_experts > 0:
                self.specific_experts = nn.ModuleList([nn.ModuleList([DCNv2(
                    input_dim=input_dim,
                    model_structure=dcnv2_model_structure,
                    stacked_dnn_hidden_units=expert_hidden_units,
                    parallel_dnn_hidden_units=expert_hidden_units,
                    dnn_activations=hidden_activations,
                    num_cross_layers=dcnv2_num_cross_layers,
                    net_dropout=net_dropout,
                    batch_norm=batch_norm
                ) for _ in range(self.num_specific_experts)]) for _ in range(self.num_tasks)])
        else:
            raise NotImplementedError("expert_type={} not supported!".format(expert_type))
                
        def get_output_dim(i, num_tasks, num_treatments, num_outcomes, num_specific_experts, num_outcome_shared_experts, num_treatment_shared_experts, num_shared_experts):
            if i < num_tasks:
                return num_specific_experts + num_outcome_shared_experts + num_treatment_shared_experts + num_shared_experts
            elif num_tasks <= i < num_tasks + num_treatments:
                return self.num_outcomes * num_specific_experts + num_outcome_shared_experts + num_shared_experts
            elif num_tasks + num_treatments <= i < num_tasks + num_treatments + num_outcomes:
                return self.num_treatments * num_specific_experts + num_treatment_shared_experts + num_shared_experts
            else:
                return self.num_tasks * num_specific_experts + self.num_treatments * num_outcome_shared_experts + self.num_outcomes * num_treatment_shared_experts + num_shared_experts
        self.gate = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                output_dim=get_output_dim(i, self.num_tasks, self.num_treatments, self.num_outcomes, num_specific_experts, num_outcome_shared_experts, num_treatment_shared_experts, num_shared_experts),
                hidden_units=gate_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            ) for i in range(self.num_tasks + self.num_treatments + self.num_outcomes + 1)
        ])
        self.gate_activation = get_activation('softmax')
        
    def forward(self, x, require_gate=False, require_treatment_shared=False):
        """
        Forward pass that can optionally return gate matrices and treatment shared outputs
        Args:
            x: Input tensor list
            require_gate: If True, return gate matrices
            require_treatment_shared: If True, return treatment shared expert outputs
        """
        specific_expert_outputs = []
        outcome_shared_expert_outputs = [] 
        treatment_shared_expert_outputs = []
        shared_expert_outputs = []
        treatment_shared_outputs = None
        gates = []
        
        # specific experts
        if self.num_specific_experts > 0:
            for i in range(self.num_tasks):
                task_expert_outputs = []
                for j in range(self.num_specific_experts):
                    task_expert_outputs.append(self.specific_experts[i][j](x[i]))
                specific_expert_outputs.append(task_expert_outputs)
        # for outcome_shared_experts
        for i in range(self.num_treatments):
            outcome_shared_expert_output_in = []
            for j in range(self.num_outcome_shared_experts):
                outcome_shared_expert_output_in.append(self.outcome_shared_experts[i][j](x[i]))
            outcome_shared_expert_outputs.append(outcome_shared_expert_output_in)
        # for treatment_shared_experts
        for i in range(self.num_outcomes):
            treatment_shared_expert_output_in = []
            for j in range(self.num_treatment_shared_experts):
                treatment_shared_expert_output_in.append(self.treatment_shared_experts[i][j](x[i]))
            treatment_shared_expert_outputs.append(treatment_shared_expert_output_in)
        # shared experts 
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))
        
        if require_treatment_shared and self.num_treatment_shared_experts > 0:
            # Print debug info
            #print(f"Number of outcomes: {self.num_outcomes}")
            #print(f"Number of treatment shared experts per outcome: {self.num_treatment_shared_experts}")
            
            treatment_shared_outputs = []
            for outcome_idx in range(self.num_outcomes):
                outcome_input = x[outcome_idx]
                for expert in self.treatment_shared_experts[outcome_idx]:
                    expert_output = expert(outcome_input)
                    treatment_shared_outputs.append(expert_output)
            
            treatment_shared_outputs = torch.stack(treatment_shared_outputs)
            
            # Shape should be [total_experts, batch_size, hidden_dim]
            # where total_experts = num_outcomes * num_treatment_shared_experts
            #print(f"Treatment shared outputs shape: {treatment_shared_outputs.shape}")
        
        # gate 
        cgc_outputs = [] 
        for i in range(self.num_tasks + self.num_treatments + self.num_outcomes + 1):
            if i < self.num_tasks:
                # For specific experts
                expert_outputs = []
                if self.num_specific_experts > 0:
                    expert_outputs.extend(specific_expert_outputs[i])
                expert_outputs.extend(outcome_shared_expert_outputs[i%self.num_treatments])
                expert_outputs.extend(treatment_shared_expert_outputs[i//self.num_treatments])
                expert_outputs.extend(shared_expert_outputs)
                gate_input = torch.stack(expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            elif self.num_tasks <= i < self.num_tasks + self.num_treatments:
                # For outcome shared experts
                expert_outputs = []
                if self.num_specific_experts > 0:
                    expert_outputs.extend(self.flatten_2d_list(specific_expert_outputs[j] for j in self.get_col_k_indices(i - self.num_tasks, self.num_outcomes, self.num_treatments)))
                expert_outputs.extend(outcome_shared_expert_outputs[i - self.num_tasks])
                expert_outputs.extend(shared_expert_outputs)
                gate_input = torch.stack(expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            elif self.num_tasks + self.num_treatments <= i < self.num_tasks + self.num_treatments + self.num_outcomes:
                # For treatment shared experts
                expert_outputs = []
                if self.num_specific_experts > 0:
                    expert_outputs.extend(self.flatten_2d_list(specific_expert_outputs[j] for j in self.get_row_k(i - self.num_tasks - self.num_treatments, self.num_outcomes, self.num_treatments)))
                expert_outputs.extend(treatment_shared_expert_outputs[i - self.num_tasks - self.num_treatments])
                expert_outputs.extend(shared_expert_outputs)
                gate_input = torch.stack(expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            else:
                # For shared experts
                expert_outputs = []
                if self.num_specific_experts > 0:
                    expert_outputs.extend(self.flatten_2d_list(specific_expert_outputs))
                expert_outputs.extend(self.flatten_2d_list(outcome_shared_expert_outputs))
                expert_outputs.extend(self.flatten_2d_list(treatment_shared_expert_outputs))
                expert_outputs.extend(shared_expert_outputs)
                gate_input = torch.stack(expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[-1]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
        
        if require_gate:
            return cgc_outputs, gates
        elif require_treatment_shared:
            return cgc_outputs, treatment_shared_outputs
        else:
            return cgc_outputs
        
    def get_col_k_indices(self, k, M, N):
        return [i for i in range(k, M*N, N)]
    
    def get_row_k(self, k, M, N):
        return [i for i in range(k*N, (k+1)*N)]
    
    def flatten_2d_list(self, nested_list):
        return [item for sublist in nested_list for item in sublist]

    def get_gate_matrix(self, x):
        """
        Get gate matrix for all tasks
        Args:
            x: Input tensor list
        Returns:
            gate_matrix: Tensor of shape [num_outcomes, num_treatments, num_experts]
        """
        with torch.no_grad():
            # Run forward pass to get gates
            outputs, gates = self.forward(x, require_gate=True)
            task_gates = gates[:self.num_tasks]  # Get only task-specific gates
            
            # shape: [num_outcomes, num_treatments, num_experts]
            gate_matrix = torch.stack(task_gates).reshape(self.num_outcomes, self.num_treatments, -1)
            
            return outputs, gate_matrix

    def forward_with_treatment_shared(self, inputs):
        """Forward pass that also returns treatment shared expert outputs"""
        specific_expert_outputs = []
        outcome_shared_expert_outputs = [] 
        treatment_shared_expert_outputs = []
        shared_expert_outputs = []
        
        # specific experts
        if self.num_specific_experts > 0:
            for i in range(self.num_tasks):
                task_expert_outputs = []
                for j in range(self.num_specific_experts):
                    task_expert_outputs.append(self.specific_experts[i][j](inputs[i]))
                specific_expert_outputs.append(task_expert_outputs)
        
        # outcome shared experts
        for i in range(self.num_treatments):
            outcome_shared_expert_output_in = []
            for j in range(self.num_outcome_shared_experts):
                outcome_shared_expert_output_in.append(self.outcome_shared_experts[i][j](inputs[i]))
            outcome_shared_expert_outputs.append(outcome_shared_expert_output_in)
        
        # treatment shared experts
        treatment_shared_outputs = None
        if self.num_treatment_shared_experts > 0:
            for i in range(self.num_outcomes):
                treatment_shared_expert_output_in = []
                for j in range(self.num_treatment_shared_experts):
                    treatment_shared_expert_output_in.append(self.treatment_shared_experts[i][j](inputs[i]))
                treatment_shared_expert_outputs.append(treatment_shared_expert_output_in)
            
            # Stack all treatment shared outputs for domain adaptation
            treatment_shared_outputs = torch.stack([
                expert(inputs[self.num_specific_experts + self.num_shared_experts + self.num_outcome_shared_experts + i])
                for i, expert in enumerate(self.treatment_shared_experts[0])  # Use first outcome's experts
            ])
        
        # shared experts
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](inputs[-1]))
        
        # gate
        cgc_outputs = []
        for i in range(self.num_tasks + self.num_treatments + self.num_outcomes + 1):
            if i < self.num_tasks:
                # For specific experts
                expert_outputs = []
                if self.num_specific_experts > 0:
                    expert_outputs.extend(specific_expert_outputs[i])
                expert_outputs.extend(outcome_shared_expert_outputs[i%self.num_treatments])
                expert_outputs.extend(treatment_shared_expert_outputs[i//self.num_treatments])
                expert_outputs.extend(shared_expert_outputs)
                gate_input = torch.stack(expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](inputs[i]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            elif self.num_tasks <= i < self.num_tasks + self.num_treatments:
                # For outcome shared experts
                expert_outputs = []
                if self.num_specific_experts > 0:
                    expert_outputs.extend(self.flatten_2d_list(specific_expert_outputs[j] for j in self.get_col_k_indices(i - self.num_tasks, self.num_outcomes, self.num_treatments)))
                expert_outputs.extend(outcome_shared_expert_outputs[i - self.num_tasks])
                expert_outputs.extend(shared_expert_outputs)
                gate_input = torch.stack(expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](inputs[i]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            elif self.num_tasks + self.num_treatments <= i < self.num_tasks + self.num_treatments + self.num_outcomes:
                # For treatment shared experts
                expert_outputs = []
                if self.num_specific_experts > 0:
                    expert_outputs.extend(self.flatten_2d_list(specific_expert_outputs[j] for j in self.get_row_k(i - self.num_tasks - self.num_treatments, self.num_outcomes, self.num_treatments)))
                expert_outputs.extend(treatment_shared_expert_outputs[i - self.num_tasks - self.num_treatments])
                expert_outputs.extend(shared_expert_outputs)
                gate_input = torch.stack(expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](inputs[i]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            else:
                # For shared experts
                expert_outputs = []
                if self.num_specific_experts > 0:
                    expert_outputs.extend(self.flatten_2d_list(specific_expert_outputs))
                expert_outputs.extend(self.flatten_2d_list(outcome_shared_expert_outputs))
                expert_outputs.extend(self.flatten_2d_list(treatment_shared_expert_outputs))
                expert_outputs.extend(shared_expert_outputs)
                gate_input = torch.stack(expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](inputs[-1]))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
        
        return cgc_outputs, treatment_shared_outputs


def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation


def get_output_activation(task):
    if task == "binary_classification":
        return nn.Sigmoid()
    elif task == "regression":
        return nn.Identity()
    else:
        raise NotImplementedError("task={} is not supported.".format(task))
    

# Positional Encoding for Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# Transformer-based Block
class Transformer_Block(nn.Module):
    """
    A Transformer-based block that, like MLP_Block, can optionally apply 
    an output activation. For gating scenarios, output_activation can be:
      - 'softmax': apply softmax over last dimension
      - 'top_k': apply top-K gating and then normalize
      - None: no activation
    
    If 'top_k' is selected, `top_k` must be provided.
    """
    def __init__(self, 
                 input_dim, 
                 d_model=64, 
                 nhead=4, 
                 num_layers=2, 
                 dim_feedforward=128, 
                 dropout=0.1,
                 output_dim=None,
                 output_activation=None,
                 top_k=None):
        super(Transformer_Block, self).__init__()
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.top_k = top_k

        # Project input into d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, 
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # If output_dim is specified, project Transformer output to output_dim
        if output_dim is not None:
            self.output_proj = nn.Linear(d_model, output_dim)
        else:
            self.output_proj = None

        # Validate top_k usage
        if self.output_activation == 'top_k':
            assert self.top_k is not None, "When using top_k activation, top_k parameter must be provided."
            assert output_dim is not None, "When using top_k, output_dim must match the number of experts."
            assert self.top_k <= output_dim, "top_k cannot be greater than output_dim."

    def forward(self, x):
        # x: [batch, input_dim]
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        x = self.input_proj(x)  # [batch, 1, d_model]
        x = self.pos_encoder(x) # [batch, 1, d_model]
        x = self.transformer_encoder(x) # [batch, 1, d_model]

        x = x.squeeze(1)  # [batch, d_model]

        # If output_dim is specified, project to [batch, output_dim]
        if self.output_proj is not None:
            x = self.output_proj(x)

        # Apply output activation if specified
        if self.output_activation == 'softmax':
            # Apply softmax over last dimension
            x = F.softmax(x, dim=1)
        elif self.output_activation == 'top_k':
            # Apply top-K gating
            # x: [batch, output_dim]
            # Find top_k indices
            top_k_values, top_k_indices = torch.topk(x, self.top_k, dim=1)
            # Create a mask to zero out all but top_k
            mask = torch.zeros_like(x)
            mask.scatter_(1, top_k_indices, top_k_values)
            # Normalize the selected values to sum to 1
            x = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)

        # If output_activation is None or anything else, do nothing
        return x


class TreatmentAwareUnit(nn.Module):
    def __init__(
        self,
        feature_dim,
        treatment_dim,
        tie_hidden_units=[32],
        hidden_activations="ELU",
        dropout_rates=0,
        batch_norm=False,
        tie_output_dim=None,
    ):
        super(TreatmentAwareUnit, self).__init__()
        self.feature_dim = feature_dim
        self.treatment_dim = treatment_dim
        self.dropout_rates = dropout_rates
        self.batch_norm = batch_norm
        self.tie_output_dim = (
            tie_output_dim if tie_output_dim is not None else treatment_dim
        )

        # Self-attention network (capturing feature relation)
        self.WQ = nn.Linear(feature_dim, feature_dim)
        self.WK = nn.Linear(feature_dim, feature_dim)
        self.WV = nn.Linear(feature_dim, feature_dim)
        self.WP = nn.Linear(feature_dim, feature_dim)  # Output projection

        # Treatment Information Extractor (TIE)
        self.tie = MLP_Block(
            input_dim=treatment_dim,
            output_dim=self.tie_output_dim,
            hidden_units=tie_hidden_units,
            hidden_activations=hidden_activations,
            output_activation=None,
            dropout_rates=dropout_rates,
            batch_norm=batch_norm,
        )

    def forward(self, feature, treatment):
        # Feature self-attention
        Q = self.WQ(feature)
        K = self.WK(feature)
        V = self.WV(feature)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.feature_dim**0.5
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_feature = torch.matmul(attention_weights, V)
        attended_feature = self.WP(attended_feature)

        # Treatment information extractor
        treatment_rep = self.tie(treatment)
        treatment_aware_feature = (
            attended_feature * treatment_rep
        )  # Element-wise product

        return treatment_aware_feature


class TreatmentEnhancedGate(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_units=[32],
        hidden_activations="ELU",
        dropout_rates=0,
        batch_norm=False,
    ):
        super(TreatmentEnhancedGate, self).__init__()
        self.Wb = MLP_Block(
            input_dim=feature_dim,
            output_dim=feature_dim,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            output_activation=nn.Sigmoid(),
            dropout_rates=dropout_rates,
            batch_norm=batch_norm,
        )

    def forward(self, initial_features, treatment_aware_features):
        gate_weights = self.Wb(initial_features)  # Bit-level weights
        enhanced_features = (initial_features * (1 - gate_weights)) + (
            treatment_aware_features * gate_weights
        )
        return enhanced_features


class TaskEnhancedGate(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=[32],
        hidden_activations="ELU",
        dropout_rates=0,
        batch_norm=False,
        treatment_enhanced_dim=None,
        num_heads=2,
    ):
        super(TaskEnhancedGate, self).__init__()
        self.treatment_enhanced_dim = (
            treatment_enhanced_dim if treatment_enhanced_dim is not None else input_dim
        )
        self.num_heads = num_heads

        # Task representation embedding
        self.task_query_proj = nn.Linear(input_dim, self.treatment_enhanced_dim)

        # Multi-head attention with proper dimensions
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.treatment_enhanced_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )

        self.prior_proj = nn.Linear(
            self.treatment_enhanced_dim, self.treatment_enhanced_dim
        )
        self.mlp = MLP_Block(
            input_dim=self.treatment_enhanced_dim,
            output_dim=self.treatment_enhanced_dim,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            output_activation=nn.Sigmoid(),
            dropout_rates=dropout_rates,
            batch_norm=batch_norm,
        )

    def forward(self, task_emb, enhanced_features):
        # task_emb: [batch, num_outcomes, num_treatments, task_dim]
        # enhanced_features: [batch, num_outcomes, num_treatments, feature_dim]

        # Reshape for attention
        batch_size, num_outcomes, num_treatments, dim = enhanced_features.shape
        flat_shape = (batch_size * num_outcomes * num_treatments, 1, dim)

        # Use reshape instead of view for better memory layout handling
        query = self.task_query_proj(task_emb.reshape(-1, dim)).reshape(*flat_shape)
        key_value = enhanced_features.reshape(-1, dim).unsqueeze(1)

        # Apply attention
        attn_out, _ = self.multihead_attn(query, key_value, key_value)
        attn_out = attn_out.squeeze(1)

        # Project and generate scaling factors
        prior = self.prior_proj(attn_out)
        scale = self.mlp(prior)

        return scale.reshape(batch_size, num_outcomes, num_treatments, -1)


class MultiTaskTreatmentPrediction(nn.Module):
    def __init__(
        self,
        input_dim,
        num_treatments,
        num_outcomes,
        hidden_units=[32],
        hidden_activations="ELU",
        dropout_rates=0,
        batch_norm=False,
    ):
        super().__init__()
        self.num_treatments = num_treatments
        self.num_outcomes = num_outcomes

        # Single prediction tower for each outcome
        self.towers = nn.ModuleList(
            [
                MLP_Block(
                    input_dim=input_dim,
                    output_dim=1,  # Single output per outcome
                    hidden_units=hidden_units,
                    hidden_activations=hidden_activations,
                    output_activation=None,
                    dropout_rates=dropout_rates,
                    batch_norm=batch_norm,
                )
                for _ in range(num_outcomes)
            ]
        )

    def forward(self, x):
        # x: [batch_size, num_outcomes, num_treatments, input_dim]
        batch_size, num_outcomes, num_treatments, dim = x.shape

        # Process each outcome separately
        predictions = []
        for i in range(self.num_outcomes):
            # Get data for current outcome
            outcome_data = x[:, i, :, :]  # [batch_size, num_treatments, input_dim]
            outcome_data = outcome_data.reshape(
                -1, dim
            )  # [batch_size*num_treatments, input_dim]

            # Use corresponding tower
            pred = self.towers[i](outcome_data)  # [batch_size*num_treatments, 1]
            pred = pred.squeeze(-1)  # [batch_size*num_treatments]
            pred = pred.reshape(
                batch_size, num_treatments
            )  # [batch_size, num_treatments]
            predictions.append(pred)

        # Stack predictions for all outcomes
        predictions = torch.stack(
            predictions, dim=1
        )  # [batch_size, num_outcomes, num_treatments]

        return predictions
