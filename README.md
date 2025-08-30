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

epoch = 100, dataset 7530 images, 1500 id, batch size = 32

| NN | Backbone | Loss | Acc  | Err Acc|
|----|------|-----|-----|----|
|DNNet|ResNet50|Arcface| 87.5%| 85.7% |
|DNNet|ResNet50|Arc+Focal| 85.9%|  83.6% |
|DNNet|ResNet50|Magface| 86.3%|  83.5%|
|DNNet|ResNet50|Mag+Focal| 86.1%| 83.6%|

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
|DNNet|ResNet50|Arcface| %| % |
|DNNet|ResNet50|Arc+Focal| %|  % |
|DNNet|ResNet50|Magface| %|  %|
|DNNet|ResNet50|Mag+Focal| %| %|