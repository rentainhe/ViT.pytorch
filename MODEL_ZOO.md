## Results
To verify that the converted model weight is correct, we simply compare it with the author's experimental results. We trained using mixed precision, and `--fp16_opt_level` was set to O2.

## CIFAR
### imagenet-21k
* [**tensorboard**](https://tensorboard.dev/experiment/Oz9GmmQIQCOEr4xbdr8O3Q)

|    model     |  dataset  | resolution | acc(official) | acc(this repo) |  time   |
|:------------:|:---------:|:----------:|:-------------:|:--------------:|:-------:|
|   ViT-B_16   | CIFAR-10  |  224x224   |       -       |     0.9908     | 3h 13m  |
|   ViT-B_16   | CIFAR-10  |  384x384   |    0.9903     |     0.9906     | 12h 25m |
|   ViT_B_16   | CIFAR-100 |  224x224   |       -       |     0.923      |  3h 9m  |
|   ViT_B_16   | CIFAR-100 |  384x384   |    0.9264     |     0.9228     | 12h 31m |
| R50-ViT-B_16 | CIFAR-10  |  224x224   |       -       |     0.9892     | 4h 23m  |
| R50-ViT-B_16 | CIFAR-10  |  384x384   |     0.99      |     0.9904     | 15h 40m |
| R50-ViT-B_16 | CIFAR-100 |  224x224   |       -       |     0.9231     | 4h 18m  |
| R50-ViT-B_16 | CIFAR-100 |  384x384   |    0.9231     |     0.9197     | 15h 53m |
|   ViT_L_32   | CIFAR-10  |  224x224   |       -       |     0.9903     | 2h 11m  |
|   ViT_L_32   | CIFAR-100 |  224x224   |       -       |     0.9276     |  2h 9m  |
|   ViT_H_14   | CIFAR-100 |  224x224   |       -       |      WIP       |         |


### imagenet-21k + imagenet2012
* [**tensorboard**](https://tensorboard.dev/experiment/CXOzjFRqTM6aLCk0jNXgAw/#scalars)

|    model     |  dataset  | resolution |  acc   |
|:------------:|:---------:|:----------:|:------:|
| ViT-B_16-224 | CIFAR-10  |  224x224   |  0.99  |
| ViT_B_16-224 | CIFAR-100 |  224x224   | 0.9245 |
|   ViT-L_32   | CIFAR-10  |  224x224   | 0.9903 |
|   ViT-L_32   | CIFAR-100 |  224x224   | 0.9285 |


### shorter train
* In the experiment below, we used a resolution size (224x224).
* [**tensorboard**](https://tensorboard.dev/experiment/lpknnMpHRT2qpVrSZi10Ag/#scalars)

|  upstream   |  model   |  dataset  | total_steps /warmup_steps | acc(official) | acc(this repo) |
|:-----------:|:--------:|:---------:|:-------------------------:|:-------------:|:--------------:|
| imagenet21k | ViT-B_16 | CIFAR-10  |          500/100          |    0.9859     |     0.9859     |
| imagenet21k | ViT-B_16 | CIFAR-10  |         1000/100          |    0.9886     |     0.9878     |
| imagenet21k | ViT-B_16 | CIFAR-100 |          500/100          |    0.8917     |     0.9072     |
| imagenet21k | ViT-B_16 | CIFAR-100 |         1000/100          |    0.9115     |     0.9216     |

## ImageNet2012