# Pretrained Models

This file documents the experimental results of models whose architectures are searched by MMNas. All the architectures can be found in `./arch` and we refer the corresponding models as **mmnas_vqa/vgd/itm**. The checkpoint files of pretrained models are also provided in this document. You may download these files, place them into `./logs/ckpts/` and run the `run_vqa/vgd/itm.py` to reproduce the following results.

## VQA

Then we trained a [mmnas_vqa](arch/mmnas_vqa.json) on the 'train+val+vg' split and evaluated the model on the 'val' split. **You can download the pretrained model at [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EW-97Xbq_z1Cq8lL1O5GHLQBH7UQLomFOqBplFL9bf83EA?e=Jwyf9V).**

| Model                                                                                  | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) | 
|:--------------------------------------------------------------------------------------:|:-------:|:-----------:|:----------:|:----------:|:---------:|
| [mcan-small](arch/mcan.json) | 1e-4    | 70.69       | 87.08      | 53.16      | 60.66     |
| mmnas_vqa | 1e-4    | 71.25       | 87.20      | 55.63      | 61.15     |

## VGD

For each of three visual grounding datasets, we trained a [mmnas_vgd](./arch/mmnas_itm.json), and got following results.

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

The [mmnas_itm](./arch/mmnas_itm.json) got the following results. **You can download the pretrained model at [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EYGlpkid3SJHoRUCsPy0x0sBPp3U5-8hLke7OJb9WGXNRw?e=fMsOap).**

<table>
<tr><th> Text Retrival </th><th> Image Retrival</th></tr>
<tr><td>

| R@1    | R@5    | R@10   |
| :-----: | :-----: | :-----: |
| 77.30 | 93.50 | 97.10 |

</td><td>

| R@1    | R@5    | R@10   |
| :-----: | :-----: | :-----: |
| 60.88 | 84.86 | 90.40 |
</td></tr> 
</table>