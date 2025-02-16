# WinCLIP
This is an unofficial implementation of [WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf) in [AnomalyCLIP](https://arxiv.org/abs/2310.18961)

The implementation of CLIP is based on [open_clip](https://github.com/mlfoundations/open_clip)
## Updates

- **03.20.2024**: Update the 2-shot, 4-shot, and 8-shot results of VisA.

  
## Performance evaluation
### Zero-shot
#### [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| carpet     |       90.9 |    33.9 |    26   |    66.3 |       99.3 |    97.8 |    99.8 |
| bottle     |       85.7 |    49.4 |    49.8 |    69.9 |       98.6 |    97.6 |    99.5 |
| hazelnut   |       95.7 |    39.1 |    33.3 |    81.3 |       92.3 |    88.6 |    96   |
| leather    |       95.5 |    30.8 |    20.5 |    86   |      100   |   100   |   100   |
| cable      |       61.3 |    12.2 |     6.2 |    39.4 |       85   |    84.8 |    89.8 |
| capsule    |       87   |    14.3 |     8.6 |    63.8 |       68.7 |    93.5 |    90.5 |
| grid       |       79.4 |    13.7 |     5.7 |    49.3 |       99.2 |    98.2 |    99.7 |
| pill       |       72.7 |    11.8 |     7   |    66.9 |       81.5 |    91.6 |    96.4 |
| transistor |       83.7 |    27   |    20.2 |    45.5 |       89.1 |    80   |    84.9 |
| metal_nut  |       49.3 |    23.8 |    10.8 |    39.7 |       96.2 |    95.3 |    99.1 |
| screw      |       91.1 |    11.3 |     5.4 |    70.2 |       71.7 |    85.9 |    87.7 |
| toothbrush |       86.2 |    10.5 |     5.5 |    67.9 |       85.3 |    88.9 |    94.5 |
| zipper     |       91.7 |    27.8 |    19.4 |    72   |       91.2 |    93.4 |    97.5 |
| tile       |       79.1 |    30.8 |    21.2 |    54.5 |       99.9 |    99.4 |   100   |
| wood       |       85.1 |    35.4 |    32.9 |    56.3 |       97.6 |    95.2 |    99.3 |
| mean       |       82.3 |    24.8 |    18.2 |    61.9 |       90.4 |    92.7 |    95.6 |

#### VisA
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| candle     |       87   |     8.9 |     2.3 |    77.7 |       94.9 |    90.6 |    95.4 |
| capsules   |       80   |     4.2 |     1.4 |    39.4 |       79.4 |    80.5 |    87.9 |
| cashew     |       84.8 |     9.6 |     4.8 |    78.4 |       91.2 |    88.9 |    96   |
| chewinggum |       95.4 |    31.5 |    24   |    69.6 |       95.5 |    93.8 |    98.2 |
| fryum      |       87.7 |    16.2 |    11.1 |    74.4 |       73.6 |    80   |    86.9 |
| macaroni1  |       50.3 |     0.1 |     0   |    24.7 |       79   |    74.2 |    80   |
| macaroni2  |       44.7 |     0.1 |     0   |     8   |       67.1 |    68.8 |    65.1 |
| pcb1       |       38.7 |     0.9 |     0.4 |    20.7 |       72.1 |    70.2 |    73   |
| pcb2       |       58.7 |     1.5 |     0.4 |    20.6 |       47   |    67.1 |    46.1 |
| pcb3       |       76   |     2.1 |     0.7 |    43.7 |       63.9 |    67.6 |    63   |
| pcb4       |       91.4 |    24.6 |    15.5 |    74.5 |       74.2 |    75.7 |    70.1 |
| pipe_fryum |       83.6 |     8.3 |     4.4 |    80.3 |       67.8 |    80.3 |    82.1 |
| mean       |       73.2 |     9   |     5.4 |    51   |       75.5 |    78.2 |    78.7 |
### Few-shot
#### MVTec AD (1-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| carpet     |       99.1 |    66.1 |    69.4 |    95.9 |      100   |    99.4 |   100   |
| bottle     |       94.3 |    60.9 |    64.9 |    85.1 |       99.4 |    98.4 |    99.8 |
| hazelnut   |       98.5 |    58.9 |    61.1 |    93.4 |       98   |    95.6 |    99   |
| leather    |       99.2 |    45.4 |    39.3 |    97.8 |      100   |    99.5 |   100   |
| cable      |       86.9 |    28.7 |    22.8 |    65   |       89.2 |    86.3 |    93.4 |
| capsule    |       96.4 |    32.3 |    24.7 |    89.7 |       83.5 |    92.4 |    96.3 |
| grid       |       94.1 |    28.7 |    19.1 |    82.1 |       99.6 |    99.1 |    99.9 |
| pill       |       92.4 |    36.1 |    28.7 |    89.8 |       89.6 |    93.3 |    98   |
| transistor |       90   |    41.2 |    41.1 |    67.5 |       89.6 |    80.9 |    85.7 |
| metal_nut  |       78.5 |    36.5 |    28.7 |    75.3 |       98.2 |    97.4 |    99.6 |
| screw      |       95.9 |    23.5 |    14.4 |    84.5 |       81.5 |    86.8 |    93.1 |
| toothbrush |       96   |    33.6 |    26.3 |    82.8 |       91.4 |    90.6 |    96.6 |
| zipper     |       97   |    46.5 |    40.8 |    90.5 |       86.4 |    90.3 |    95.8 |
| tile       |       91.7 |    53.5 |    46.2 |    77.5 |      100   |    99.4 |   100   |
| wood       |       94.5 |    56.4 |    59.4 |    84.5 |       99   |    96.8 |    99.7 |
| mean       |       93.6 |    43.2 |    39.1 |    84.1 |       93.7 |    93.7 |    97.1 |
#### MVTec AD (2-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| carpet     |       99   |    64.5 |    68.2 |    95.6 |       99.8 |    98.9 |    99.9 |
| bottle     |       94.8 |    62.4 |    66.3 |    85.9 |       99.6 |    98.4 |    99.9 |
| hazelnut   |       98.7 |    61.6 |    63.9 |    93.6 |       97.9 |    95.7 |    98.9 |
| leather    |       99.2 |    45.2 |    39   |    97.9 |       99.9 |    99.5 |   100   |
| cable      |       88.8 |    31.5 |    25.3 |    72.5 |       91   |    89.5 |    94   |
| capsule    |       95.6 |    23.1 |    11.9 |    86.8 |       66   |    92.6 |    88.1 |
| grid       |       94.8 |    30.3 |    20.7 |    83.9 |       99.4 |    99.1 |    99.8 |
| pill       |       92.8 |    39.5 |    32.9 |    90.4 |       92.9 |    95.3 |    98.6 |
| transistor |       89.8 |    41   |    40.4 |    66.7 |       89.5 |    79.2 |    85.6 |
| metal_nut  |       76.7 |    35.2 |    26.8 |    73.8 |       98.5 |    98.4 |    99.7 |
| screw      |       96.7 |    25.6 |    18   |    87.5 |       82.9 |    86.9 |    93.5 |
| toothbrush |       96.4 |    36.6 |    29.9 |    82   |       93.3 |    92.1 |    97.6 |
| zipper     |       97.2 |    50   |    43.9 |    91.1 |       95.2 |    94.8 |    98.7 |
| tile       |       92   |    53.9 |    46.4 |    78   |       99.9 |    99.4 |   100   |
| wood       |       94.5 |    56.2 |    58.5 |    86   |       99.5 |    98.3 |    99.8 |
| mean       |       93.8 |    43.8 |    39.5 |    84.8 |       93.7 |    94.5 |    96.9 |
#### MVTec AD (4-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| carpet     |       99   |    65.1 |    68.9 |    95.5 |       99.9 |    99.4 |   100   |
| bottle     |       94.4 |    62.1 |    65.7 |    85.2 |       99.4 |    97.6 |    99.8 |
| hazelnut   |       98.5 |    60.7 |    62.5 |    92.8 |       97.6 |    95   |    98.8 |
| leather    |       99.3 |    45.4 |    39.3 |    97.8 |      100   |   100   |   100   |
| cable      |       89   |    31.7 |    25.8 |    71.4 |       89.6 |    88.4 |    92.9 |
| capsule    |       97.2 |    35.7 |    27.9 |    91.1 |       86.5 |    94   |    96.9 |
| grid       |       95.1 |    30   |    22   |    84   |       99.7 |    98.2 |    99.9 |
| pill       |       93   |    40.9 |    34.4 |    90.9 |       92.4 |    94.1 |    98.5 |
| transistor |       89.4 |    40.6 |    39.2 |    65.5 |       90.4 |    80.4 |    87.3 |
| metal_nut  |       80.2 |    38   |    31.1 |    78   |       99.3 |    98.4 |    99.8 |
| screw      |       96   |    22   |    15.1 |    85   |       81.4 |    89.1 |    91.6 |
| toothbrush |       98.2 |    55.1 |    50.8 |    88.6 |       98.1 |    96.7 |    99.3 |
| zipper     |       97.4 |    51.3 |    46.2 |    91.2 |       95.5 |    94.8 |    98.8 |
| tile       |       91.7 |    53.1 |    45.3 |    77.7 |      100   |    99.4 |   100   |
| wood       |       94.5 |    56.6 |    59.3 |    86.6 |       99.3 |    97.5 |    99.8 |
| mean       |       94.2 |    45.9 |    42.2 |    85.4 |       95.3 |    94.9 |    97.6 |
#### MVTec AD (8-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| carpet     |       98.9 |    64.8 |    68.5 |    94.7 |       99.8 |    99.4 |    99.9 |
| bottle     |       94.8 |    61.9 |    65.9 |    85.3 |       99.5 |    98.4 |    99.8 |
| hazelnut   |       98.8 |    62.3 |    64.3 |    93.8 |       98.4 |    96.4 |    99.1 |
| leather    |       99.2 |    44.8 |    38.2 |    97.4 |      100   |   100   |   100   |
| cable      |       89.7 |    32.9 |    26.8 |    73.9 |       91.4 |    89.8 |    94   |
| capsule    |       96.9 |    34.6 |    27   |    90.6 |       85.5 |    93.6 |    96.6 |
| grid       |       95.6 |    31   |    23.6 |    85.9 |       99.6 |    99.1 |    99.9 |
| pill       |       93.4 |    42.4 |    35.9 |    91.4 |       93.3 |    94.4 |    98.7 |
| transistor |       90.8 |    42.6 |    41.6 |    68.6 |       91.1 |    81.4 |    86.8 |
| metal_nut  |       79.9 |    37.9 |    30.6 |    77.9 |       99.2 |    98.9 |    99.8 |
| screw      |       96.8 |    17.9 |    12.4 |    86   |       80.4 |    88.8 |    91.1 |
| toothbrush |       98.3 |    55.4 |    51.8 |    89.4 |       98.9 |    96.7 |    99.6 |
| zipper     |       97.4 |    51.2 |    45.5 |    91.6 |       97.6 |    96   |    99.4 |
| tile       |       91.6 |    53   |    45   |    76.6 |      100   |    99.4 |   100   |
| wood       |       94.8 |    56.1 |    59.2 |    87.3 |       99.6 |    98.4 |    99.9 |
| mean       |       94.5 |    45.9 |    42.4 |    86   |       95.6 |    95.4 |    97.6 |

#### VisA (1-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| candle     |       94.8 |    16.8 |     8   |    90.5 |       96.4 |    91   |    96.9 |
| capsules   |       95.9 |    32.3 |    22.4 |    65.4 |       80.2 |    80.2 |    87.8 |
| cashew     |       96.6 |    32.3 |    22.2 |    90.3 |       95.4 |    91.8 |    97.9 |
| chewinggum |       99   |    57   |    54.6 |    85.9 |       97.7 |    94.8 |    99   |
| fryum      |       94.4 |    32.4 |    26   |    86   |       87.7 |    86.6 |    94.4 |
| macaroni1  |       91.5 |    13   |     4   |    78.9 |       85.6 |    78.8 |    87.7 |
| macaroni2  |       91.6 |     4   |     0.9 |    72.1 |       75.4 |    72.4 |    74.6 |
| pcb1       |       96   |    16.1 |     7.3 |    76.8 |       85.6 |    83.7 |    84.5 |
| pcb2       |       91.5 |     6.6 |     2.9 |    66.4 |       59.6 |    67.9 |    57   |
| pcb3       |       92.9 |    13.4 |     7.6 |    77.6 |       68.9 |    71.2 |    68.9 |
| pcb4       |       95.3 |    22.4 |    15.9 |    82.4 |       85.5 |    79   |    85.6 |
| pipe_fryum |       96.4 |    28   |    18.9 |    94.1 |       88   |    85.6 |    94.2 |
| mean       |       94.7 |    22.9 |    15.9 |    80.5 |       83.8 |    81.9 |    85.7 |
#### VisA (2-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| candle     |       95.5 |    18.2 |     8.7 |    91.2 |       96   |    90.9 |    96.7 |
| capsules   |       95.8 |    31.8 |    22.5 |    65.4 |       82.4 |    83   |    89.8 |
| cashew     |       96.9 |    34.2 |    24   |    89.1 |       93.4 |    89.4 |    97   |
| chewinggum |       99   |    57.4 |    56   |    86.5 |       98.2 |    95.9 |    99.2 |
| fryum      |       95   |    35.5 |    27.9 |    86.1 |       84.7 |    84.8 |    92.7 |
| macaroni1  |       93.8 |    13.4 |     4.5 |    83.8 |       88   |    81.5 |    89.6 |
| macaroni2  |       91.4 |     3.7 |     0.8 |    70.2 |       73.3 |    72.6 |    73.4 |
| pcb1       |       96.2 |    17   |     8.1 |    77.7 |       85.4 |    83.7 |    83.6 |
| pcb2       |       92.1 |     7.1 |     3.2 |    65.9 |       58   |    69.2 |    57.7 |
| pcb3       |       93.8 |    19.2 |    10.3 |    80.7 |       72   |    70.3 |    70.5 |
| pcb4       |       95.9 |    21.9 |    14.1 |    84.1 |       79.3 |    80.7 |    70.1 |
| pipe_fryum |       96.2 |    28   |    18.7 |    93.9 |       89.7 |    87.9 |    95   |
| mean       |       95.1 |    23.9 |    16.6 |    81.2 |       83.4 |    82.5 |    84.6 |
#### VisA (4-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| candle     |       95.8 |    19.2 |     9.3 |    91.2 |       96.7 |    90.7 |    97.2 |
| capsules   |       96.1 |    32.2 |    23.1 |    66.6 |       82.2 |    81.7 |    89.3 |
| cashew     |       96.7 |    32.4 |    22.2 |    89.4 |       93.4 |    89.3 |    97   |
| chewinggum |       99   |    57.2 |    55.5 |    85.9 |       98   |    96.5 |    99.2 |
| fryum      |       94.9 |    34.3 |    27.8 |    87.5 |       87.1 |    86.6 |    93.8 |
| macaroni1  |       93.9 |    14.2 |     4.8 |    83.6 |       89.1 |    82.9 |    90.7 |
| macaroni2  |       90   |     4.8 |     1   |    68.9 |       76   |    73.2 |    76.1 |
| pcb1       |       96.2 |    16.9 |     8   |    77.1 |       86.8 |    84.2 |    84.2 |
| pcb2       |       91.7 |     9.8 |     4.5 |    65.4 |       59.6 |    68.8 |    59.4 |
| pcb3       |       94.6 |    23.3 |    13.2 |    81.1 |       69.9 |    70.7 |    68.7 |
| pcb4       |       96.7 |    30.8 |    23.4 |    86.1 |       80.7 |    76.6 |    79.5 |
| pipe_fryum |       96.3 |    28.1 |    19.1 |    94.4 |       89.8 |    87.1 |    95   |
| mean       |       95.2 |    25.3 |    17.7 |    81.4 |       84.1 |    82.4 |    85.8 |
#### VisA (8-shot)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| candle     |       95.9 |    19.2 |     9.4 |    91.1 |       96.9 |    91.5 |    97.3 |
| capsules   |       96.2 |    32.2 |    23.2 |    66.9 |       82.6 |    83   |    89.3 |
| cashew     |       96.9 |    34.2 |    24.4 |    89.5 |       95.6 |    92.2 |    98   |
| chewinggum |       99   |    56.7 |    54.5 |    85.8 |       97.9 |    96   |    99.1 |
| fryum      |       95   |    35.4 |    28   |    88.2 |       89.7 |    86.7 |    95.3 |
| macaroni1  |       93.9 |    13.7 |     4.7 |    82.9 |       89.6 |    82   |    90.8 |
| macaroni2  |       89   |     3.8 |     0.6 |    68.3 |       76.7 |    73.5 |    76.1 |
| pcb1       |       96.2 |    17   |     8.3 |    76.8 |       87.4 |    84.7 |    85.4 |
| pcb2       |       92.5 |    10.8 |     4.9 |    66   |       63.7 |    70.1 |    61   |
| pcb3       |       95.2 |    23.9 |    14.7 |    81.7 |       76.1 |    74.8 |    74   |
| pcb4       |       97.2 |    33.5 |    25.8 |    88   |       84.6 |    81.8 |    83.1 |
| pipe_fryum |       96.5 |    29.3 |    19.7 |    94.3 |       91.2 |    89.3 |    95.7 |
| mean       |       95.3 |    25.8 |    18.2 |    81.6 |       86   |    83.8 |    87.1 |

## Quick start
Zero-shot anomaly detection 
```sh
bash zero_shot.sh
```
Few-shot anomaly detection 
```sh
bash few_shot.sh
```


## BibTex Citation

If you find this paper and repository useful, please cite our paper.

```
@article{zhou2024anomalyclip,
  title={AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection},
  author={Zhou, Qihang and Pang, Guansong and Tian, Yu and He, Shibo and Chen, Jiming},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
