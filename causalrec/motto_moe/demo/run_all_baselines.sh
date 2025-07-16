# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
#!/bin/bash

# Script to run all baseline experiments in experiments/synthetic/

BASE_DIR="../experiments/synthetic"

echo "==================================================="
echo "Starting All Synthetic Baseline Experiments"
echo "==================================================="

echo "---------------------------------------------"
echo "Running MOTTO baseline ($BASE_DIR/baselines_MOTTO.py)..."
echo "---------------------------------------------"
python $BASE_DIR/baselines_MOTTO.py

echo ""
echo "---------------------------------------------------"
echo "Running MOTTO_DA baseline ($BASE_DIR/baselines_MOTTO_DA.py)..."
echo "---------------------------------------------------"
python $BASE_DIR/baselines_MOTTO_DA.py

echo ""
echo "-----------------------------------------------------------"
echo "Running MTML CFRNet MMD ($BASE_DIR/baselines_MTML_CFRNet_mmd.py)..."
echo "-----------------------------------------------------------"
python $BASE_DIR/baselines_MTML_CFRNet_mmd.py

echo ""
echo "-----------------------------------------------------------"
echo "Running MTML CFRNet Wass ($BASE_DIR/baselines_MTML_CFRNet_wass.py)..."
echo "-----------------------------------------------------------"
python $BASE_DIR/baselines_MTML_CFRNet_wass.py

echo ""
echo "-------------------------------------------------------"
echo "Running MTML DragonNet ($BASE_DIR/baselines_MTML_DragonNet.py)..."
echo "-------------------------------------------------------"
python $BASE_DIR/baselines_MTML_DragonNet.py

echo ""
echo "-----------------------------------------------------"
echo "Running MTML S-Learner ($BASE_DIR/baselines_MTML_Slearner.py)..."
echo "-----------------------------------------------------"
python $BASE_DIR/baselines_MTML_Slearner.py

echo ""
echo "-----------------------------------------------------"
echo "Running MTML TARNet ($BASE_DIR/baselines_MTML_TARNet.py)..."
echo "-----------------------------------------------------"
python $BASE_DIR/baselines_MTML_TARNet.py

echo ""
echo "-----------------------------------------------------"
echo "Running MTML T-Learner ($BASE_DIR/baselines_MTML_Tlearner.py)..."
echo "-----------------------------------------------------"
python $BASE_DIR/baselines_MTML_Tlearner.py

echo ""
echo "----------------------------------------------------------"
echo "Running MTML VanillaMoE ($BASE_DIR/baselines_MTML_VanillaMoE.py)..."
echo "----------------------------------------------------------"
python $BASE_DIR/baselines_MTML_VanillaMoE.py

echo ""
echo "==================================================="
echo "All Synthetic Baseline Experiments Completed."
echo "===================================================" 
