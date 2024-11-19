# MiniTorch Module 3

## Timing differences (prove that GPU operations lead to speedups on large matrix operations)

Timing summary
Size: 64
    fast: 0.00359
    gpu: 0.00680
Size: 128
    fast: 0.01679
    gpu: 0.01562
Size: 256
    fast: 0.10427
    gpu: 0.07443
Size: 512
    fast: 1.08795
    gpu: 0.22570
Size: 1024
    fast: 7.98068
    gpu: 1.08512

## SIMPLE on GPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.1`

| Epoch | Loss | Correct | Seconds per epoch |
|-------|------|---------|-------------------|
| 0     | 6.3237 | 41      | 1.9933            |
| 10    | 1.9080 | 48      | 1.9463            |
| 20    | 0.3265 | 48      | 1.8684            |
| 30    | 0.3960 | 50      | 1.9524            |
| 40    | 0.2769 | 50      | 1.8692            |
| 50    | 0.0131 | 50      | 1.9633            |
| 60    | 0.0375 | 48      | 1.8655            |
| 70    | 0.4200 | 50      | 1.9429            |
| 80    | 0.0267 | 50      | 1.9573            |
| 90    | 0.1953 | 50      | 1.9573            |

## SIMPLE on CPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.1`

| Epoch | Loss | Correct | Seconds per epoch |
|-------|------|---------|-------------------|
| 0     | 5.3371 | 38      | 0.5354            |
| 10    | 1.8547 | 49      | 0.1142            |
| 20    | 0.8975 | 49      | 0.1239            |
| 30    | 1.0269 | 50      | 0.2219            |
| 40    | 0.2756 | 50      | 0.1187            |
| 50    | 0.3368 | 50      | 0.1140            |
| 60    | 0.3647 | 50      | 0.1113            |
| 70    | 0.0841 | 50      | 0.1112            |
| 80    | 0.2813 | 50      | 0.1127            |
| 90    | 0.1136 | 50      | 0.1136            |

## SPLIT on GPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05`

| Epoch | Loss | Correct | Seconds per epoch |
|-------|------|---------|-------------------|
| 0     | 11.7159 | 37      | 1.9775            |
| 10    | 4.2385  | 39      | 1.8452            |
| 20    | 2.7342  | 45      | 1.9120            |
| 30    | 1.7094  | 46      | 1.8245            |
| 40    | 1.6292  | 47      | 1.9075            |
| 50    | 1.7776  | 49      | 1.8244            |
| 60    | 0.6428  | 47      | 1.9963            |
| 70    | 0.7133  | 50      | 1.8989            |
| 80    | 0.5307  | 49      | 1.8398            |
| 90    | 0.9868  | 50      | 1.8398            |

## SPLIT on CPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`

| Epoch | Loss | Correct | Seconds per epoch |
|-------|------|---------|-------------------|
| 0     | 5.8618 | 30      | 0.5386            |
| 10    | 6.4110 | 35      | 0.1917            |
| 20    | 5.0855 | 41      | 0.1602            |
| 30    | 4.5376 | 47      | 0.1134            |
| 40    | 4.8999 | 44      | 0.1131            |
| 50    | 2.5158 | 49      | 0.1132            |
| 60    | 2.7788 | 47      | 0.1145            |
| 70    | 2.5462 | 48      | 0.1124            |
| 80    | 2.7626 | 50      | 0.1172            |
| 90    | 1.1745 | 49      | 0.1172            |

## XOR on GPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.08`

| Epoch | Loss | Correct | Seconds per epoch |
|-------|------|---------|-------------------|
| 0     | 14.1093 | 29      | 1.9467            |
| 10    | 2.4116  | 46      | 1.9167            |
| 20    | 5.3343  | 46      | 1.9097            |
| 30    | 3.6910  | 44      | 1.8270            |
| 40    | 0.7536  | 46      | 1.9059            |
| 50    | 1.0581  | 48      | 1.8177            |
| 60    | 4.1409  | 44      | 1.9096            |
| 70    | 0.8909  | 50      | 1.8328            |
| 80    | 1.7812  | 50      | 1.9079            |
| 90    | 1.6082  | 50      | 1.8269            |

## XOR on CPU, small (100)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.08`

| Epoch | Loss | Correct | Seconds per epoch |
|-------|------|---------|-------------------|
| 0     | 6.8074 | 34      | 0.6575            |
| 10    | 4.9934 | 45      | 0.1155            |
| 20    | 3.1434 | 46      | 0.1164            |
| 30    | 2.0709 | 49      | 0.1121            |
| 40    | 2.5962 | 47      | 0.1115            |
| 50    | 2.2167 | 47      | 0.1110            |
| 60    | 1.8634 | 49      | 0.1116            |
| 70    | 0.8398 | 48      | 0.1112            |
| 80    | 1.8089 | 47      | 0.1133            |
| 90    | 1.3300 | 50      | 0.1265            |

## THE BIGGER MODEL: SPLIT on GPU, big (500)

`!cd $DIR; PYTHONPATH=/content/$DIR python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 300 --DATASET split --RATE 0.05`

| Epoch | Loss | Correct | Seconds per epoch |
|-------|------|---------|-------------------|
| 0     | 12.3483 | 30      | 2.0010            |
| 10    | 3.1553  | 45      | 1.9975            |
| 20    | 2.4289  | 46      | 1.8969            |
| 30    | 0.1419  | 47      | 2.0043            |
| 40    | 1.2024  | 46      | 1.9131            |
| 50    | 0.6739  | 50      | 1.9714            |
| 60    | 0.7581  | 50      | 1.9761            |
| 70    | 0.4917  | 50      | 1.8990            |
| 80    | 0.2741  | 50      | 2.0014            |
| 90    | 0.4053  | 50      | 2.0014            |

## `python project/parallel_check.py` output

For improved readability, please go to the following pastebin to see the output:

https://pastebin.com/B3hs6YQi
