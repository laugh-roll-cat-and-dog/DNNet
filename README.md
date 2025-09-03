# Accuracy

(Not use) epoch = 32, dataset 2000 pairs(negative pairs not the same ;-;)

| NN | Backbone | Loss | Acc |
|----|------|-----|-----|
|DNNet|ResNet50|Contrastive|78.7%|
|DNNet|ResNet152|Contrastive| %|
|DNNet|ResNet50|Arcface|83.1%|
|DNNet|ResNet152|Arcface| %|
|DNNet|ResNet50|Con+Arc| 87.1%|
|DNNet|ResNet152|Con+Arc|85.6%|
|DNNet|ResNet50|Soft triplet| 86.4% |
|DNNet|ResNet152|Soft triplet| 84.1% |
|DNNet|ResNet50|Con+Soft| 85.1% |
|DNNet|ResNet152|Con+Soft| 84.0% |

epoch = 200, dataset 10340 pairs, 350 id, + pairs == - pairs

| NN | Backbone | Loss | Acc |
|----|------|-----|-----|
|DNNet|ResNet50|Arcface| 86.2%|
|DNNet|ResNet50|Con+Arc|84.8%|
|DNNet|ResNet50|Soft triplet| 85.9% |
|DNNet|ResNet50|Con+Soft| 82.2% |

# Accuracy classification

epoch = 200, dataset 8400 pairs, 350 id, (+ pairs) * 3 == - pairs

| NN | Backbone | Loss | Acc | Class Acc | Err Acc|
|----|------|-----|-----|----|---|
|DNNet|ResNet50|Arcface| 86.4%| 65.0% | 83.7% |
|DNNet|ResNet50|Con+Arc|84.1%|54.5%| 81.5% |
|DNNet|ResNet50|Magface| 83.8%| 58.9% | 81.2% |
|DNNet|ResNet50|Con+Mag|81.7%|45.0%| 78.7%|
|DNNet|ResNet50|Mag+Focal|86.0%|59.8%| 84.0%|

# Test batch size

epoch = 100, dataset 7530 images, 1500 id, batch size = 16

| NN | Backbone | Loss | Acc | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Arcface| 85.9%| 84.6% |
|DNNet|ResNet50|Arc+Soft| 87.4%| 86.6% |
|DNNet|ResNet50|Arc+Focal| 86.3%|  84.0% |
|DNNet|ResNet50|Arc+Focal+Soft| 85.9%|  84.0% |
|DNNet|ResNet50|Magface| 86.0%|  84.0%|
|DNNet|ResNet50|Mag+Soft| 84.9%| 83.4% |
|DNNet|ResNet50|Mag+Focal| 86.2%| 84.0%|
|DNNet|ResNet50|Mag+Focal+Soft| 86.3%|  82.0% |

epoch = 100, dataset 7530 images, 1500 id, batch size = 32

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Arcface| 87.5%| 85.7% |
|DNNet|ResNet50|Arc+Soft| 86.4%| 85.2% |
|DNNet|ResNet50|Arc+Focal| 85.9%|  83.6% |
|DNNet|ResNet50|Arc+Focal+Soft| 86.2%|  83.1% |
|DNNet|ResNet50|Magface| 86.3%|  83.5%|
|DNNet|ResNet50|Mag+Soft| 86.5%| 83.6% |
|DNNet|ResNet50|Mag+Focal| 86.1%| 83.6%|
|DNNet|ResNet50|Mag+Focal+Soft| 85.4%|  83.0% |

epoch = 100, dataset 7530 images, 1500 id, batch size = 64

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Arcface| 87.1%| 84.2% |
|DNNet|ResNet50|Arc+Focal| 85.5%|  84.2% |
|DNNet|ResNet50|Magface| 85.2%|  81.7%|
|DNNet|ResNet50|Mag+Focal| 86.7%| 82.5%|

epoch = 100, dataset 7530 images, 1500 id, batch size = 128

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Arcface| 85.2%| 82.9% |
|DNNet|ResNet50|Arc+Focal| 85.7%|  82.4% |
|DNNet|ResNet50|Magface| 84.0%|  81.3%|
|DNNet|ResNet50|Mag+Focal| 84.5%| 81.6%|

# Test lambda_g in Magface

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g = 20

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Magface| 86.0%|  84.0%|
|DNNet|ResNet50|Mag+Soft| 84.9%| 83.4% |
|DNNet|ResNet50|Mag+Focal| 86.2%| 84.0%|
|DNNet|ResNet50|Mag+Focal+Soft| 86.3%|  82.0% |

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g = 15

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Magface| 86.5%|  83.8%|
|DNNet|ResNet50|Mag+Soft| 85.8%| 81.2% |
|DNNet|ResNet50|Mag+Focal| 85.3%| 82.7%|
|DNNet|ResNet50|Mag+Focal+Soft| 85.2%|  81.4% |

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g = 10

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Magface| 85.3%|  83.3%|
|DNNet|ResNet50|Mag+Soft| 85.2%| 81.8% |
|DNNet|ResNet50|Mag+Focal| 85.3%| 82.5%|
|DNNet|ResNet50|Mag+Focal+Soft| 85.1%| 81.7% |

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g = 5

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Magface| 86.6%|  84.6%|
|DNNet|ResNet50|Mag+Soft| 85.1%| 81.8% |
|DNNet|ResNet50|Mag+Focal| 86.5%| 85.1%|
|DNNet|ResNet50|Mag+Focal+Soft| 84.9%| 81.1% |

epoch = 100, dataset 7530 images, 1500 id, batch size = 16, lambda_g = 5

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Magface| 86.6%|  85.1%|
|DNNet|ResNet50|Mag+Soft|86.2%| 82.4% |
|DNNet|ResNet50|Mag+Focal| 85.6%| 82.7%|
|DNNet|ResNet50|Mag+Focal+Soft| 84.5%| 82.5% |

# Test ResNet-152

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g(magface) = 5

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet152|Arcface| 84.8%|  82.5%|
|DNNet|ResNet152|Arc+Soft| 84.6%| 82.9% |
|DNNet|ResNet152|Arc+Focal|85.4 %| 82.2% |
|DNNet|ResNet152|Arc+Focal+Soft| 86.6%| 83.4% |
|DNNet|ResNet152|Magface| 85.4%|  82.8%|
|DNNet|ResNet152|Mag+Soft| 85.3%| 81.5% |
|DNNet|ResNet152|Mag+Focal| 84.5%| 82.4%|
|DNNet|ResNet152|Mag+Focal+Soft| 84.1%| 80.9% |

epoch = 200, dataset 7530 images, 1500 id, batch size = 32, lambda_g(magface) = 5

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet152|Arcface| 85.0%|  83.3%|
|DNNet|ResNet152|Arc+Soft| 86.0%| 83.4% |
|DNNet|ResNet152|Arc+Focal|84.9%| 83.5% |
|DNNet|ResNet152|Arc+Focal+Soft| 83.8%|81.5% |
|DNNet|ResNet152|Magface| 85.1%|  82.3%|
|DNNet|ResNet152|Mag+Soft| 84.0%| 80.3% |
|DNNet|ResNet152|Mag+Focal|84.6%| 81.5%|
|DNNet|ResNet152|Mag+Focal+Soft| 83.1%| 80.9% |