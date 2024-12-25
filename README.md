# Data Alignment Based Adversarial Defense Benchmark for EEG-based BCIs
This project evaluated some typical adversarial defense approaches in BCIs based on data alignment.

## Data alignment
We performed Euclidean Alignment (EA) for data alignment. There are many proper ways to perform EA. We used the code in [data_align.py](https://github.com/chamwen/LSFT/blob/main/data_align.py) to perform EA. 

An example of us performing online EA is [here](https://github.com/xqchen914/EEGAdversarialBenchmark/blob/main/utils/data_loader.py#L83).

An example of us performing offline EA is [here](https://github.com/xqchen914/EEGAdversarialBenchmark/blob/main/MIget2014001.py#L125).

## Defenses
Following defenses were evaluated:
- Normal training (NT): ``` NT.py ```
- Adversarial training (AT): ``` AT.py ```
- TRADES: ``` TRADES.py ```
- SCR: ```SCR.py```
- Stochastic activation pruning (SAP): ``` SAP.py ```
- Input transformation (IT): ``` IT.py ```
- Random self ensemble (RSE): ``` RES.py ```
- Self ensemble adversarial training (SEAT): ``` SEAT.py ```
- Self ensemble adversarial training with data augmentation (SEAT-DA): ``` SEATDA.py ```

## Attacks
Two white-box attacks and two black-box attacks were used to evaluate defenses, which can be found in file ``` attack_lib.py ``` and ``` autoattack ``` folder. 

## Evaluation
The file ``` eval.py ``` can be used for evaluation after model trained with defense. For example, the evaluation of EEGNet trained with AT against $\ell_{\infty}$ untargeted FGSM attacks with adversarial perturbation amplitude 0.03 in within-session setting is as follows:  
``` python3 eval.py --model=EEGNet --defense=AT_0.03 --distance=inf --target=0 --attack=FGSM --setup=within_sess```


## Acknowledgements
Credit of the base framework goes to [`bci_adv_defense`](https://github.com/lbinmeng/bci_adv_defense) project. 