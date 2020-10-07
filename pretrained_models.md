# Pretrained Models

This file documents the experiment results of models whose architectures are searched by MMNasNet. All the architectures can be found in `./logs/ckpts/arch` and we refer the corresponding models as **mmnas_vqa/vgd/itm**. The checkpoint files of trained models are also provided in this document. You may download these files, place them into `./logs/ckpts/` and run the `run_vqa/vgd/itm.py` to reproduce the following results.

## VQA

### train -> val

We trained a [mmnas_vqa](logs/ckpts/arch/mmnas_vqa.json) on the 'train' split and evaluated the model on the 'val' split.

| Model                                                                                  | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) |
|:--------------------------------------------------------------------------------------:|:-------:|:-----------:|:----------:|:----------:|:---------:|
| [mcan-large](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mcan_large.yml) | 7e-5    | 67.50       | 85.14      | 49.66      | 58.80     |
| mmnas_vqa | 1.2e-4    | 67.79       | 85.02      | 52.25      | 58.80     |

### train+val+vg -> test-dev

Then we trained a [mmnas_vqa](logs/ckpts/arch/mmnas_vqa.json) on the 'train+val+vg' split and evaluated the model on the 'val' split. **You can download the trained model at [here](./).**

| Model                                                                                  | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) | 
|:--------------------------------------------------------------------------------------:|:-------:|:-----------:|:----------:|:----------:|:---------:|
| [mcan-large](https://github.com/MILVLG/openvqa/tree/master/configs/vqa/mcan_large.yml) | 5e-5    | 70.82       | 87.19      | 52.56      | 60.98     |
| mmnas_vqa | 1e-4    | 71.24       | 87.11      | 56.15      | 61.08     |

## VGD

For each of three visual grounding datasets, we trained a [mmnas_vgd](./logs/ckpts/arch/mmnas_itm.json), and got following results.

<table>
<tr><th> RefCOCO </th><th> RefCOCO+ </th><th> RefCOCORg </th></tr>
<tr><td>

| Val     | TestA   | TestB |
| :-------: | :-------: | :-----: |
| 83.66\% | 87.25\% | 78.78 |
</td><td>

| Val     | TestA   | TestB   |
| :-------: | :-------: | :-----: |
| 74.48\% | 81.00\% | 65.15\% |
</td><td>

| Val     | Test    |
| :-------: | :-------: |
| 74.59\% | 75.42\% |
</td></tr> 
<tr> 
    <td align="center"> <a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/Ea0S0zcCV45GhVqWOeW0PHoBKkh6NwJRyVpCh8-cmpwFOA?e=IKJ09r">model</a> </td> 
    <td align="center"> <a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVU2eXV6RLpCiHrDMOBtNLwB1hN0Kn88pC1lKEXCDUfZGQ?e=9yM1GF">model</a> </td> 
    <td align="center"> <a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVU2eXV6RLpCiHrDMOBtNLwB1hN0Kn88pC1lKEXCDUfZGQ?e=9yM1GF">model</a> </td> 
</tr>
</table>

## ITM

The [mmnas_itm](./logs/ckpts/arch/mmnas_itm.json) got the following results. **You can download the trained model at [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EYGlpkid3SJHoRUCsPy0x0sBPp3U5-8hLke7OJb9WGXNRw?e=fMsOap).**

<table>
<tr><th> Text Retrival </th><th> Text Retrival</th></tr>
<tr><td>

| @1    | @5    | @10   |
| :-----: | :-----: | :-----: |
| 77.30 | 93.50 | 97.10 |

</td><td>

| @1    | @5    | @10   |
| :-----: | :-----: | :-----: |
| 60.88 | 84.86 | 90.40 |
</td></tr> 
</table>