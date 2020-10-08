# Pretrained Models

This document contains the models to reproduce of results by using the are searched archtectures from MMnas. The prototxt files for mmnas architecture for different tasks can be found in the `arch` folder. We provide pretrained models on different tasks to reproduce the results reported in our paper. You can download these ckpt files, place them at `logs/ckpts/`, and run the `run_[vqa/vgd/itm].py` to evaluate the performance. 

## VQA

We train the [mmnas_vqa](arch/mmnas_vqa.json) on the 'train+val+vg' split and evaluated the model on the 'test-dev' split. **You can download the pretrained model at [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EW-97Xbq_z1Cq8lL1O5GHLQBH7UQLomFOqBplFL9bf83EA?e=Jwyf9V).** For comparsion, we also provide the results from previous state-of-the-art [mcan](https://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.html) model

| Model                                                                                  | Base lr | Overall (%) | Yes/No (%) | Number (%) | Other (%) | 
|:--------------------------------------------------------------------------------------:|:-------:|:-----------:|:----------:|:----------:|:---------:|
| [mcan](arch/mcan.json) | 1e-4    | 70.69       | 87.08      | 53.16      | 60.66     |
| mmnas | 1e-4    | 71.25       | 87.20      | 55.63      | 61.15     |

## VGD

We use the same [mmnas_vgd](./arch/mmnas_itm.json) archtecture for all the three datasets and then train the model for each dataset independently.

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

Using the [mmnas_itm](./arch/mmnas_itm.json) archtecture, we obtain the model to report the following results. The pretrained model can be downloaded [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EYGlpkid3SJHoRUCsPy0x0sBPp3U5-8hLke7OJb9WGXNRw?e=fMsOap).

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
