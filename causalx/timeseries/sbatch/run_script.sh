
cd CausalTransformer_rebuttal/

source venv/bin/activate

mlflow server --port=5000 & 

PYTHONPATH=. python3 runnables/train_msm.py +dataset=cancer_sim +backbone=msm exp.seed=10 dataset.coeff=5.0 dataset.random=False dataset.noise_percentage=0.0 ++dataset.treatment_mode=multilabel

PYTHONPATH=. python3 runnables/train_rmsn.py +dataset=cancer_sim +backbone=rmsn +backbone/rmsn_hparams/cancer_sim=5.yaml dataset.coeff=5.0 dataset.random=False dataset.noise_percentage=0.00 exp.seed=10 ++dataset.treatment_mode=multilabel


PYTHONPATH=. python3 runnables/train_gnet.py +dataset=cancer_sim +backbone=gnet +backbone/gnet_hparams/cancer_sim=2.yaml exp.seed=10

PYTHONPATH=. python3 runnables/train_multi.py +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_subgroup_random=\'2\' exp.seed=10

PYTHONPATH=. python3 runnables/train_enc_dec.py +dataset=cancer_sim +backbone=crn +backbone/crn_hparams/cancer_sim_grad_reverse=\'1\' exp.seed=10

Gating purpose

Across-treatment dimension (knowledge sharing)
Sample needs different factual & alignment balance

python train.py -m hydra/sweeper/sampler=random objective=auuc_outcome_0 hydra.sweeper.n_trials=3 hydra.sweeper.sampler.seed=42 model=ple2D_DA seeds=[42,123,666]

python train.py -m hydra/sweeper/sampler=random objective=auuc_outcome_0 hydra.sweeper.n_trials=3 hydra.sweeper.sampler.seed=42 model=Vanilla_MMoE seeds=[42,123,666]




=== Epoch [2141/3000] (alpha=8) ===
[Epoch 2141] W1 Loss (cluster) = 22.8421, Source Clf (weighted) = 0.2150, Unweighted Source Clf = 0.2603, Centroid Loss = 0.9005, SNTG = 7.3842, Epoch Time = 15.25s
  Standard Accuracy on target data: 0.7378
  Balanced Accuracy on target data: 0.6775
  Best Balanced Accuracy so far: 0.7650 (achieved at epoch 1917)
  Best Standard Accuracy so far: 0.7756 (achieved at epoch 824)

======== Training Complete ========
Total epochs:                3000
Total training time:         51112.99s (~851.88 min)
Peak GPU memory usage:       54.54 MB
Best Standard Accuracy:      0.7800 (epoch 2404)
Best Balanced Accuracy:      0.7650 (epoch 1917)
Average time per epoch:          15.63s
Average forward/backward time:   1.9467s (per mini-batch)


[Epoch 3000] W1 Loss (cluster) = 22.9437, Source Clf (weighted) = 0.1534, Unweighted Source Clf = 0.2358, Centroid Loss = 0.9151, SNTG = 7.1571, Epoch Time = 15.69s
  Standard Accuracy on target data: 0.7489
  Balanced Accuracy on target data: 0.6750
  Best Balanced Accuracy so far: 0.7650 (achieved at epoch 1917)
  Best Standard Accuracy so far: 0.7800 (achieved at epoch 2404)
Adjusting learning rate of group 0 to 3.5116e-30.
Adjusting learning rate of group 0 to 3.5116e-30.
Adjusting learning rate of group 0 to 3.5116e-30.


======== Training Complete ========
Total epochs:                3000
Total training time:         51112.99s (~851.88 min)
Peak GPU memory usage:       54.54 MB
Best Standard Accuracy:      0.7800 (epoch 2404)
Best Balanced Accuracy:      0.7650 (epoch 1917)
Average time per epoch:          15.63s
Average forward/backward time:   1.9467s (per mini-batch)







======== Training Complete ========
Total iterations:            100000
Total training time:         34224.55s (~570.41 min)
Average iteration time:      0.332s
Peak GPU memory usage:       112.08 MB
Total trainable parameters:  4,657,369




https://proceedings.neurips.cc/paper_files/paper/2023/file/94ab02a30b0e4a692a42ccd0b4c55399-Paper-Conference.pdf
Efficient targeted learning of heterogeneous treatment effects for multiple subgroups
https://arxiv.org/abs/2211.14671
https://opus.lib.uts.edu.au/handle/10453/177140

https://www.internalfb.com/code/time_series_cate/src/settings


Random covariate  5 - exp.seed=10

0
judicious-fowl-963

[2024-09-23 14:44:04,759][__main__][INFO] - Test normalised RMSE (all): 0.8407027867486985; Test normalised RMSE (orig): 0.5852164076866554; Test normalised RMSE (only counterfactual): 0.7350317274118907


[2024-09-23 14:48:51,872][__main__][INFO] - Test normalised RMSE (n-step prediction): {'2-step': 0.6861786827785632, '3-step': 0.6912249618943208, '4-step': 0.7072567731551049, '5-step': 0.7253492593999921, '6-step': 0.7453193479146434}

1

bouncy-fish-337


[2024-09-23 14:49:42,817][__main__][INFO] - Test normalised RMSE (all): 0.8122460463651624; Test normalised RMSE (orig): 0.578313630868252; Test normalised RMSE (only counterfactual): 0.745538173213847


[2024-09-23 14:56:06,914][__main__][INFO] - Test normalised RMSE (n-step prediction): {'2-step': 0.6767686309857008, '3-step': 0.6969864617370649, '4-step': 0.7350357172026171, '5-step': 0.7652631552124795, '6-step': 0.7999261005516095}



2

magnificent-bear-189

[2024-09-23 14:43:50,605][__main__][INFO] - Test normalised RMSE (all): 0.8344279136077681; Test normalised RMSE (orig): 0.6034854427560356; Test normalised RMSE (only counterfactual): 0.762160214039291


[2024-09-23 14:48:57,840][__main__][INFO] - Test normalised RMSE (n-step prediction): {'2-step': 0.6934043591481271, '3-step': 0.7203979524381348, '4-step': 0.7518814950300124, '5-step': 0.7874972681156127, '6-step': 0.8187863873110225}

3
angry-chimp-716

[2024-09-23 14:44:04,436][__main__][INFO] - Test normalised RMSE (all): 0.9656389705304055; Test normalised RMSE (orig): 0.7115714691068868; Test normalised RMSE (only counterfactual): 0.9010720509609476


[2024-09-23 14:49:19,754][__main__][INFO] - Test normalised RMSE (n-step prediction): {'2-step': 0.7846452492033532, '3-step': 0.8557640609418499, '4-step': 0.9214070022004264, '5-step': 0.9675332252863277, '6-step': 1.0222612959634365}


4
nimble-squirrel-944

[2024-09-23 15:14:22,112][__main__][INFO] - Test normalised RMSE (all): 1.03233549319858; Test normalised RMSE (orig): 0.8146153878922011; Test normalised RMSE (only counterfactual): 1.0380378993988546

[2024-09-23 15:19:49,854][__main__][INFO] - Test normalised RMSE (n-step prediction): {'2-step': 1.0038685400877871, '3-step': 1.1944214801891009, '4-step': 1.36173848887031, '5-step': 1.5218286384335618, '6-step': 1.6632113192185543}





Random covariate  5 - exp.seed=101

0
fun-asp-920

Test normalised RMSE (only counterfactual): 0.7241515096099977

[2024-09-24 04:14:21,413][__main__][INFO] - Test normalised RMSE (n-step prediction): {'2-step': 0.5898487403628567, '3-step': 0.6058001782659059, '4-step': 0.624746584967353, '5-step': 0.6536618770329594, '6-step': 0.6861735178648016}

1

tasteful-panda-998

Test normalised RMSE (only counterfactual): 0.7422272428939725

[2024-09-24 04:45:04,871][__main__][INFO] - Test normalised RMSE (n-step prediction): {'2-step': 0.6095038461956122, '3-step': 0.617567326127232, '4-step': 0.6350852239794335, '5-step': 0.6544332641003467, '6-step': 0.6754228684626375}

2
debonair-yak-147
Test normalised RMSE (only counterfactual): 0.7937071642145765

[2024-09-24 05:12:49,455][__main__][INFO] - Test normalised RMSE (n-step prediction): {'2-step': 0.6137085620416736, '3-step': 0.6482151414888498, '4-step': 0.6756561927332054, '5-step': 0.6984159260021766, '6-step': 0.7258851445665725}

3


Impression

1. Send an invite 
Thank you so much again for giving me the call and sending over my letter. Made my weekend? Couple of qs on the package? Do you have some time in the next few days for a quick call? 18k

2. Directly

Can you explain more on the equity? Generic qs. Why there’s a range of the equity, more about the what is vesting schedule? Guaranteed

Basically no in the first three years

Nothing bad can happen.

1. Direct and ask, you mention there might be the option to include a sign-on bonus to the compensation? I talk to a couple of Lilly colleagues their compensation package is constructed/comes with a sign-on bonus. 
2. I’m employed at Meta, discussion maybe converting to FTE, in the range of 330k. This will help me in accepting Lilly’s offer and not moving forward with them. 15k or 20k. Thank you that’s very generous. I’m on some conversations with my other colleagues at Meta, I understand difference.
3. 10k - 20k a bit lower than what I was expecting, 30k? Wait for her to say her number

Is there anything you could do to sweeten the current offer from a financial perspective? For instance a sign-on bonus

This will help me in accepting Lilly’s offer and not moving forward with them.

There is no such requirement when I applied
https://web.archive.org/web/20240000000000*/https://cs.duke.edu/graduate/ms/ms-req


Bill of Fill

akmitchell@ncdot.gov


conda activate mtml

pip install --upgrade hydra-ax-sweeper


ValueError: mutable default <class 'hydra_plugins.hydra_ax_sweeper.config.EarlyStopConfig'> for field early_stop is not allowed: use default_factory

conda install python=3.10.14

  File "/data/home/yilingliu/miniconda3/envs/mtml/lib/python3.11/dataclasses.py", line 815, in _get_field
    raise ValueError(f'mutable default {type(f.default)} for field '
ValueError: mutable default <class 'hydra_plugins.hydra_ax_sweeper.config.EarlyStopConfig'> for field early_stop is not allowed: use default_factory

  - python=3.10.14
  - xz=5.4.6
  - python_abi=3.10=*_cpython



FACT
F

/hpc/home/yl407/.conda/envs/pytorch_env_A6000/bin

conda activate /hpc/home/yl407/carlsonlab_sync/conda_env/pytorch_env_A6000


--gres=gpu:RTX2080:1

conda activate pytorch_env_A6000


======== Training Complete ========
Total epochs:                10
Total training time:         207.75s (~3.46 min)
Peak GPU memory usage:       54.57 MB
Best Standard Accuracy:      0.4667
Best Balanced Accuracy:      0.4288
Average time per epoch:          19.19s
Average forward/backward time:   2.3977s (per mini-batch)
Training logs saved to 'training_logs.csv'.

[Final Log Summary]
epoch_times: [32.474079608917236, 17.47133779525757, 17.19221782684326, 17.828604698181152, 17.378488779067993, 17.440146684646606, 17.662049293518066, 18.461202383041382, 18.199162244796753, 17.758365154266357]
forward_backward_times: [11.968589067459106, 5.1368937492370605, 3.467263698577881, 2.9118809700012207, 2.3675427436828613, 2.2558257579803467, 2.1409153938293457, 2.184760808944702, 2.214388370513916, 2.17092227935791, 2.1389427185058594, 2.1063425540924072, 2.1428306102752686, 2.204789400100708, 2.2357780933380127, 2.256537914276123, 2.2370765209198, 2.177354574203491, 2.1803293228149414, 2.2014782428741455, 2.0997233390808105, 2.0825657844543457, 2.087843179702759, 2.125063419342041, 2.1256251335144043, 2.1446707248687744, 2.268242120742798, 2.272209644317627, 2.2838618755340576, 2.2147376537323, 2.2836906909942627, 2.2349045276641846, 2.1354339122772217, 2.1632351875305176, 2.0526466369628906, 2.1675853729248047, 2.2412540912628174, 2.1666009426116943, 2.1535141468048096, 2.2975897789001465, 2.110858917236328, 2.2543647289276123, 2.2073400020599365, 2.2672338485717773, 2.1955087184906006, 2.1859657764434814, 2.11712384223938, 2.101119041442871, 2.1222434043884277, 2.1122007369995117, 2.1048803329467773, 2.2720532417297363, 2.2532615661621094, 2.271994113922119, 2.273036479949951, 2.251821279525757, 2.283517837524414, 2.3881983757019043, 2.2729086875915527, 2.277916669845581, 2.302849769592285, 2.282278060913086, 2.300746440887451, 2.3521549701690674, 2.2917234897613525, 2.274721145629883, 2.2656898498535156, 2.307828903198242, 2.2838099002838135, 2.319031000137329, 2.271786689758301, 2.183934211730957, 2.099602222442627, 2.1867079734802246, 2.1484599113464355, 2.0949342250823975, 2.276270866394043, 2.302093029022217, 2.374852180480957, 2.274742603302002]
peak_memory_MB: 54.57470703125
standard_accuracy_history: [0.45111111111111113, 0.42, 0.4666666666666667, 0.4577777777777778, 0.4311111111111111, 0.46444444444444444, 0.4488888888888889, 0.47555555555555556, 0.4577777777777778, 0.46]
balanced_accuracy_history: [0.41999999999999993, 0.39375000000000004, 0.4287500000000001, 0.41500000000000004, 0.4, 0.41000000000000003, 0.38375, 0.37249999999999994, 0.40625, 0.39]
best_standard_accuracy: 0.4666666666666667
best_balanced_accuracy: 0.4287500000000001
total_params: 103202




yl407@dcc-carlsonlab-gpu-18. DANN.py
yl407@dcc-carlsonlab-gpu-04 wdgrl.py
yl407@dcc-carlsonlab-gpu-06 adda.py

DANN
[INFO] Feature Extractor params: 5780
[INFO] Classifier params: 16560
[INFO] Discriminator params: 17091
[INFO] Total trainable params: 39431

======== Training Complete ========
Total epochs:                30
Total training time:         888.33s (~14.81 min)
Peak GPU memory usage:       1.28 MB
Best Standard Accuracy:      0.6067 (epoch 13)
Best Balanced Accuracy:      0.6646 (epoch 21)
Training logs saved to 'dann_training_logs.csv'.


Wdgrl
[INFO] Feature Extractor params: 5780
[INFO] Discriminator params: 16560
[INFO] Critic params: 17091
[INFO] Total trainable params: 39431


======== Training Complete ========
Total epochs:                100
Total training time:         14733.25s (~245.55 min)
Peak GPU memory usage:       2.92 MB
Best Standard Accuracy:      0.2060 (epoch 24)
Best Balanced Accuracy:      0.2810 (epoch 8)
Training logs saved to 'wdgrl_training_logs.csv'.


Adda
[INFO] Source Feature Extractor params: 0
[INFO] Target Feature Extractor params: 5780
[INFO] Discriminator params: 17091
[INFO] Total trainable params: 22871


======== Training Complete ========
Total epochs:                5
Total training time:         1119.81s (~18.66 min)
Peak GPU memory usage:       3.25 MB
Best Standard Accuracy:      0.4864 (epoch 2)
Best Balanced Accuracy:      0.6150 (epoch 4)
Training logs saved to 'adda_training_logs.csv'.


DSN

[INFO] DSN Model params: 1102159
[INFO] Total trainable params: 1102159
Adjusting learning rate of group 0 to 1.0000e-02.
=== Epoch [1/100] ===

======== Training Complete ========
Total epochs:                100
Total training time:         3087.98s (~51.47 min)
Peak GPU memory usage:       15.78 MB
Best Standard Accuracy:      0.5789 (epoch 71)
Best Balanced Accuracy:      0.6467 (epoch 35)
Training logs saved to 'dsn_training_logs.csv'.
done

pixelDA

[INFO] Generator params: 475795
[INFO] Discriminator params: 1555585
[INFO] Classifier params: 1571466
[INFO] Total trainable params: 3602846

Standard Accuracy on target data: 0.8224                                                                                                  
Balanced Accuracy on target data: 0.8896

======== Training Complete ========
Total epochs:                200
Total training time:         7593.43s (~126.56 min)
Peak GPU memory usage:       99.92 MB
Best Standard Accuracy:      0.8438 (epoch 85)
Best Balanced Accuracy:      0.9016 (epoch 85)
Training logs saved to 'pixelda_training_logs.csv'.




DRANet

=== Model Parameters ===
Encoder: 57,408 parameters
Generator: 57,251 parameters
Separator: 204,928 parameters
Discriminators: 1,776,002 parameters
Task_Networks: 2,561,780 parameters
Total: 4,657,369 parameters


Evaluating M2MM...
Results for M2MM:
Standard Accuracy: 0.6158
Balanced Accuracy: 0.7664


======== Training Complete ========
Total iterations:            100000
Total training time:         34224.55s (~570.41 min)
Average iteration time:      0.332s
Peak GPU memory usage:       112.08 MB
Total trainable parameters:  4,657,369



module load CUDA/10.2



Source only imbalance New sntg 1


Per-class accuracy on target domain:
aeroplane   : 33.83%
bicycle     : 88.00%
bus         : 0.13%
car         : 93.00%
horse       : 48.86%
knife       : 8.08%
motorcycle  : 0.75%
person      : 21.21%
plant       : 19.60%
skateboard  : 85.00%
train       : 2.13%
truck       : 5.00%

Mean accuracy: 33.80%
EPOCH 001: train_loss=25.0395, train_accuracy=0.7868, val_loss=24.5059, val_accuracy=0.8794, target_accuracy=0.3380
Saving model...
                                                                                                   
Per-class accuracy on target domain:
aeroplane   : 91.73%
bicycle     : 89.00%
bus         : 0.25%
car         : 96.00%
horse       : 68.84%
knife       : 11.00%
motorcycle  : 49.69%
person      : 20.20%
plant       : 47.55%
skateboard  : 90.91%
train       : 7.65%
truck       : 1.02%

Mean accuracy: 47.82%


                                                                                                                                                   
Per-class accuracy on target domain:
aeroplane   : 96.98%
bicycle     : 67.00%
bus         : 34.17%
car         : 90.00%
horse       : 86.22%
knife       : 30.00%
motorcycle  : 85.41%
person      : 17.00%
plant       : 76.91%
skateboard  : 86.00%
train       : 86.29%
truck       : 0.00%

Mean accuracy: 63.00%
EPOCH 007: train_loss=24.2450, train_accuracy=0.9565, val_loss=24.1627, val_accuracy=0.9551, target_accuracy=0.6300
Saving model...


Per-class accuracy on target domain:
aeroplane   : 97.99%
bicycle     : 67.01%
bus         : 44.36%
car         : 88.00%
horse       : 87.50%
knife       : 32.00%
motorcycle  : 86.56%
person      : 14.00%
plant       : 81.10%
skateboard  : 85.00%
train       : 89.22%
truck       : 2.00%

Mean accuracy: 64.56%
EPOCH 010: train_loss=24.1485, train_accuracy=0.9652, val_loss=23.9764, val_accuracy=0.9597, target_accuracy=0.6456
Saving model...


                                                                                                                                                   
Per-class accuracy on target domain:
aeroplane   : 97.23%
bicycle     : 65.66%
bus         : 51.63%
car         : 87.88%
horse       : 89.94%
knife       : 23.00%
motorcycle  : 85.46%
person      : 18.00%
plant       : 83.06%
skateboard  : 88.00%
train       : 90.84%
truck       : 0.00%

Mean accuracy: 65.06%
EPOCH 013: train_loss=24.0885, train_accuracy=0.9686, val_loss=23.9676, val_accuracy=0.9655, target_accuracy=0.6506
Saving model...





Source Only Imbalance New

aeroplane   : 30.70%
bicycle     : 87.00%
bus         : 0.00%
car         : 88.00%
horse       : 61.36%
knife       : 9.09%
motorcycle  : 1.75%
person      : 25.25%
plant       : 18.09%
skateboard  : 91.00%
train       : 8.77%
truck       : 6.00%

Mean accuracy: 35.59%


Per-class accuracy on target domain:
aeroplane   : 89.60%
bicycle     : 85.00%
bus         : 0.63%
car         : 92.00%
horse       : 77.89%
knife       : 18.00%
motorcycle  : 60.73%
person      : 24.24%
plant       : 42.64%
skateboard  : 92.93%
train       : 46.30%
truck       : 3.06%

Mean accuracy: 52.75%


Per-class accuracy on target domain:
aeroplane   : 95.87%
bicycle     : 81.82%
bus         : 5.53%
car         : 91.92%
horse       : 81.81%
knife       : 29.29%
motorcycle  : 82.29%
person      : 29.00%
plant       : 56.84%
skateboard  : 92.00%
train       : 82.43%
truck       : 4.08%

Mean accuracy: 61.07%


Per-class accuracy on target domain:
aeroplane   : 96.48%
bicycle     : 80.00%
bus         : 12.58%
car         : 90.00%
horse       : 83.85%
knife       : 33.33%
motorcycle  : 82.52%
person      : 28.28%
plant       : 59.19%
skateboard  : 90.00%
train       : 90.87%
truck       : 2.02%

Mean accuracy: 62.43%

Per-class accuracy on target domain:
aeroplane   : 97.37%
bicycle     : 74.75%
bus         : 31.91%
car         : 85.00%
horse       : 85.87%
knife       : 30.00%
motorcycle  : 88.07%
person      : 26.00%
plant       : 59.25%
skateboard  : 88.89%
train       : 94.20%
truck       : 3.06%

Mean accuracy: 63.70%



Per-class accuracy on target domain:
aeroplane   : 96.61%
bicycle     : 71.72%
bus         : 55.53%
car         : 81.00%
horse       : 87.92%
knife       : 38.00%
motorcycle  : 92.08%
person      : 26.26%
plant       : 69.42%
skateboard  : 82.00%
train       : 96.37%
truck       : 2.00%

Mean accuracy: 66.58%



Per-class accuracy on target domain:
aeroplane   : 97.86%
bicycle     : 72.00%
bus         : 60.62%
car         : 82.00%
horse       : 87.81%
knife       : 32.32%
motorcycle  : 92.17%
person      : 31.00%
plant       : 68.67%
skateboard  : 88.89%
train       : 95.73%
truck       : 3.03%

Mean accuracy: 67.68%



Per-class accuracy on target domain:
aeroplane   : 97.61%
bicycle     : 72.73%
bus         : 62.81%
car         : 84.85%
horse       : 88.55%
knife       : 33.00%
motorcycle  : 91.85%
person      : 31.00%
plant       : 73.02%
skateboard  : 89.00%
train       : 96.36%
truck       : 5.00%

Mean accuracy: 68.82%
EPOCH 013: train_loss=2.5448, train_accuracy=0.9720, val_loss=2.5380, val_accuracy=0.9688, target_accuracy=0.6882
Saving model...



Per-class accuracy on target domain:
aeroplane   : 97.36%
bicycle     : 66.00%
bus         : 63.41%
car         : 83.67%
horse       : 88.60%
knife       : 46.00%
motorcycle  : 93.33%
person      : 28.28%
plant       : 75.38%
skateboard  : 89.00%
train       : 94.72%
truck       : 1.00%

Mean accuracy: 68.90%
EPOCH 020: train_loss=2.5235, train_accuracy=0.9749, val_loss=2.5099, val_accuracy=0.9761, target_accuracy=0.6890
Saving model...


Per-class accuracy on target domain:
aeroplane   : 96.47%
bicycle     : 69.00%
bus         : 67.25%
car         : 80.81%
horse       : 89.63%
knife       : 47.47%
motorcycle  : 92.09%
person      : 29.29%
plant       : 76.72%
skateboard  : 88.89%
train       : 94.99%
truck       : 3.00%

Mean accuracy: 69.63%
EPOCH 022: train_loss=2.5153, train_accuracy=0.9764, val_loss=2.5084, val_accuracy=0.9755, target_accuracy=0.6963
Saving model...

                                                                                                                                                   
Per-class accuracy on target domain:
aeroplane   : 97.86%
bicycle     : 64.00%
bus         : 69.71%
car         : 78.79%
horse       : 89.81%
knife       : 59.00%
motorcycle  : 92.71%
person      : 33.00%
plant       : 78.67%
skateboard  : 89.90%
train       : 95.61%
truck       : 1.02%

Mean accuracy: 70.84%
EPOCH 024: train_loss=2.5098, train_accuracy=0.9777, val_loss=2.5064, val_accuracy=0.9750, target_accuracy=0.7084
Saving model...








                                                                                                                                                 
Per-class accuracy on target domain:
aeroplane   : 97.40%
bicycle     : 31.74%
bus         : 14.67%
car         : 49.02%
horse       : 83.13%
knife       : 11.44%
motorcycle  : 98.15%
person      : 29.71%
plant       : 74.79%
skateboard  : 85.40%
train       : 96.95%
truck       : 28.11%

Mean accuracy: 58.38%


Per-class accuracy on target domain:
aeroplane   : 98.07%
bicycle     : 28.65%
bus         : 22.01%
car         : 52.10%
horse       : 87.32%
knife       : 13.37%
motorcycle  : 98.25%
person      : 29.97%
plant       : 76.29%
skateboard  : 81.97%
train       : 97.30%
truck       : 17.35%

Mean accuracy: 58.55%


Mean accuracy: 58.55%
EPOCH 026: train_loss=0.0397, train_accuracy=0.9889, val_loss=0.0375, val_accuracy=0.9886, target_accuracy=0.5855
Saving model...
                                                                                                                                                 
Per-class accuracy on target domain:
aeroplane   : 98.53%
bicycle     : 32.99%
bus         : 22.07%
car         : 53.40%
horse       : 84.79%
knife       : 15.27%
motorcycle  : 98.62%
person      : 23.49%
plant       : 77.26%
skateboard  : 81.77%
train       : 96.40%
truck       : 19.58%

Mean accuracy: 58.68%



Per-class accuracy on target domain:
aeroplane   : 98.00%
bicycle     : 32.94%
bus         : 17.91%
car         : 51.28%
horse       : 86.76%
knife       : 15.01%
motorcycle  : 98.15%
person      : 30.50%
plant       : 74.94%
skateboard  : 85.13%
train       : 97.20%
truck       : 25.44%

Mean accuracy: 59.44%


Source only
Per-class accuracy on target domain:
aeroplane   : 26.50%
bicycle     : 13.04%
bus         : 0.00%
car         : 0.13%
horse       : 26.81%
knife       : 0.00%
motorcycle  : 97.37%
person      : 24.36%
plant       : 56.97%
skateboard  : 91.08%
train       : 99.40%
truck       : 26.92%

Mean accuracy: 38.55%

ELS imbalance


Source dataset size: 78165
Target dataset size: 55388




--gres=gpu:RTX2080:1

conda activate /hpc/home/yl407/carlsonlab_sync/conda_env/pytorch_env_A6000

export CUDA_HOME=/opt/apps/rhel9/cuda-12.4 
export PATH=$CUDA_HOME/bin:$PATH 
exportLD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


export PYTHONNOUSERSITE=1


Weaknesses
1. While the paper discusses the effectiveness of using Wasserstein distances, it lacks a detailed discussion on the computational analysis, especially in high-dimensional spaces where such calculations can become challenging.
2. The paper could benefit from broader comparisons with recent state-of-the-art methods. Including more diverse UDA scenarios and newer methods could help in establishing the relative performance of DARSA more comprehensively. As unsupervised domain adaptation (UDA) is an active research area, more recent baseline methods should be included. The latest baseline cited in the paper dates back to 2021.
3. The discussion on the limitations of the proposed method is somewhat limited. A more detailed exploration of scenarios where DARSA might underperform, such as extreme domain shifts or very high-dimensional data, would provide a more balanced view of the method’s applicability. One limitation I can see is the high-dimensionality challenge that may arise from using Wasserstein distances.
4. The generalization of the proposed method across different types of domain adaptation tasks (beyond those tested) is not extensively discussed. Insights into how DARSA performs with non-Gaussian distributions or with datasets having complex or non-convex structures could enhance the paper. How does DARSA perform on the VisDA-2017 dataset?
Requested Changes:
1. Including a computational analysis on the proposed DARSA. 
Actual computational time to run wasserstein? O(N^2) +alignment/no alignment sinkhorn quadratic
A100 run
1. Including more recent baseline methods for comparisons. TVT and CDAN [implementation done]
2. Including more discussion on the limitations of the proposed theoretical work and DARSA.
3. Running experiments on VisDA-2017 dataset. [add our DARSA on top of 2 other SOTAs]
TVT[2023], EUDA[2024], CDAN + DARSA
 DANN vs DARSA, done for VisDA 

Rationale for Dataset Selection:Our primary goal in choosing digit transfer tasks (e.g., MNIST, USPS, SVHN) was to validate DARSA’s theoretical contributions against well-established, interpretable benchmarks. These datasets’ simplicity and controlled conditions make it easy to visualize domain shifts and confirm DARSA’s sub-domain alignment capabilities under label shifts. Their small, manageable class sets also mirror conditions in critical application domains (e.g., medical scenarios), where accuracy and stability are paramount.
Rationale for Baseline Selection:We carefully selected baselines to demonstrate DARSA’s robustness and theoretical soundness. Alongside established domain adaptation methods such as DANN, we included top-performing methods on digit benchmarks listed on Papers with Code [1], as well as sub-domain-based approaches [2, 3] closely aligned with our research focus. This ensures that our comparisons are both comprehensive and relevant.
To stay current with the field’s rapid advancements, we have now incorporated state-of-the-art methods published after 2021, chosen for their relevance to unsupervised domain adaptation (UDA) and their performance on comparable tasks. The updated comparisons underscore DARSA’s enduring competitiveness.
https://github.com/jonas-lippl/fact
https://github.com/uta-smile/TVT

Extending to Larger and More Complex Datasets and Incorporating Newer Baselines::While digit datasets serve as a strong proof-of-concept, we have also evaluated DARSA on VisDA-2017, a large-scale benchmark with substantial domain gaps. The results (Section [X], Table [Y]) demonstrate DARSA’s ability to generalize beyond simple tasks, further confirming its practical applicability.
In addition to established digit benchmarks, we have included newer, top-performing methods on VisDA-2017 for a fair, contemporary comparison. These expanded evaluations, detailed in the revised manuscript, illustrate DARSA’s resilience and adaptability in a rapidly evolving research landscape.
[1] https://paperswithcode.com/task/domain-adaptation
[2] Deng, Zhijie, Yucen Luo, and Jun Zhu. "Cluster alignment with a teacher for unsupervised domain adaptation." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
[3] Long, Mingsheng, et al. "Conditional adversarial domain adaptation." Advances in neural information processing systems 31 (2018).


Strengths And Weaknesses:
Strengths: The authors provide a theoretical foundation for sub-domain-based domain adaptation methods, addressing gap where previous methods relied solely on empirical success, strengthening the understanding of why sub-domain alignment outperforms full-domain alignment.
The authors analyse an important failure case where the marginal weights of sub-domains differ between the source and target domains. They propose a reweighted loss function to address this issue providing a deeper understanding of when and why sub-domain methods can fail and how to fix them.
Weaknesses: While the theoretical contributions are novel, the components of DARSA- weighted classification loss, Wasserstein alignment, and clustering loss have been extensively used in previous works. The novelty lies more in the theoretical analysis rather than the components themselves. 
The theoretical analysis relies on strong assumptions such as sub-domains following Gaussian distributions, and distances between paired source-target sub-domains being smaller than non-paired distances. While these assumptions work on certain datasets, they may not hold in highly complex or noisy real-world domain, thereby weakening the theoretical guarantees.
Target sub-domains are defined using pseudo-labels generated by the target encoder and classifier, which could be noisy leading to inaccurate sub-domain definitions that may degrade the performance.
Computing the Wasserstein distance for multiple sub-domains, along with intra and inter clustering losses could introduce significant computational overhead. For large-scale datasets with many sub-domains, DARSA may be slower and less scalable compared to simpler domain adaptation methods.
This work mainly addresses sub-domain weight shifts, however it does not deeply explore other types of domain shift, such as feature shift or covariate shift, so the proposed approach might be less effective in cases where shifts occur at levels other than sub-domain weights.
Requested Changes:
Critical Adjustments: As mentioned in the weaknesses section, the theoretical results rely on strong assumptions, please provide a thorough justification for these assumptions by discussing their limitations. Include scenarios/challenges where these assumptions might fail and discuss how DARSA would perform under such conditions to ensure that the theoretical contributions are robust and widely applicable.
DARSA relies on pseudo-labels to define target sub-domains, but pseudo-labels may be noisy, particularly in the early training stages. Consider analysing the impact of pseudo-label noise on sub-domain alignment and generalization performance and incorporate techniques for mitigating the noise. 

True label vs pseudo-label compare, empirically performance good, admit limitation on generalization performance using pseudolabel future work 

Provide an analysis of the computational cost of DARSA, including time complexity and resource requirements and compare its efficiency with other domain adaptation methods on large-scale datasets. 
While the theoretical contributions are significant, the algorithmic components (weighted losses, Wasserstein alignment, clustering loss) are well-known. Can you clarify and highlight how DARSA improves over existing methods by integrating these components.
Other Adjustments: While DARSA focuses on label distribution shifts, exploring other types of domain shift such as feature shift, covariate shift would make the work more comprehensive. 
Add a short discussion on the limitations of DARSA, such as its reliance on pseudo-labels, computational costs, and the assumptions made in the theoretical analysis.


Weaknesses:
1. Confusion between class and domain
I find the paper difficult to follow due to the confusion between the concepts of 'class' and 'domain.' From my understanding, the term 'sub-domain' in the paper refers to individual classes within each domain. However, this usage is misleading, as 'sub-domain' might be interpreted as a hierarchical domain structure where a target domain is comprised of smaller sub-domains. Moreover, the paper uses 'class' and 'domain' interchangeably, contradicting literature on domain adaptation and dataset shift. I recommend referring to relevant literature (e.g., [1, 2]) to clearly define the distinction between 'domain' and 'class.'
1. Generalization bound and evaluation metric under class imbalance
In the analysis of the generalization bound, the empirical target error is defined by the empirical error on the target domain, i.e., $\epsilon_S(h, f)=\mathrm{E}_{\mathbf{x} \sim \mathcal{D}_S}[|h(\mathbf{x})-f(\mathbf{x})|]$ in [3]. However, this may not be the ideal bound under class imbalance. In the extreme imbalance setup, a trivial classifier that always assigns a class label to the majority class could lead to low empirical target error. Empirical results from the paper only report the prediction accuracy and disregard evaluations under class imbalance, e.g., balanced accuracy. I think more consideration should be given to what bound/metric to optimize.
References:
[1] Quiñonero-Candela, Joaquin, et al., eds. Dataset shift in machine learning. Mit Press, 2022.
[2] Pan, Sinno Jialin, and Qiang Yang. "A survey on transfer learning." IEEE Transactions on knowledge and data engineering 22.10 (2009): 1345-1359.
[3] Ben-David, Shai, et al. "A theory of learning from different domains." Machine learning 79 (2010): 151-175.
Requested Changes:
Please address the issues in the weakness section above.
1. Provide a clear definition of class, domain
2. Elaborate on considerations for class imbalance in generalization bound, especially on the empirical target error and its suitability under imbalance.
3. Include class imbalance metrics, e.g., balance accuracy, in empirical results and verify the model does not learn trivial solutions.
Add at least one experiment for POC. 
Class imbalance not that bad
1. Improve the writing and clarity of this paper. I find the introduction section and Fig. 1 very challenging to understand due to the confusion between class and domain.

“

(base) yl407@dcc-carlsonlab-gpu-01  ~/carlsonlab_sync/DARSA_rebuttal_tmlr/experiments/mnist_to_mnist_m $ python DARSA_tune_v2.py

yl407@dcc-carlsonlab-gpu-02  ~/carlsonlab_sync/DARSA_rebuttal_tmlr/experiments/mnist_to_mnist_m $ python DARSA_tune_save_weights.py

DARSA VisDA-2017 must!!!!


Today’s planning

3hr MMoE: run model iteration for critero
3hr DARSA finish empirical evaluation and calculate balance accuracy
3hr DARSA VisDa 2017






Average Accuracy = 0.8565
Average Class Accuracy = 0.8186
Per-Class Accuracies:
    aeroplane: 0.9775
    bicycle: 0.9100
    bus: 0.5288
    car: 0.7800
    horse: 0.9675
    knife: 0.9200
    motorcycle: 0.8950
    person: 0.8900
    plant: 0.9388
    skateboard: 0.9400
    train: 0.8963
    truck: 0.1800