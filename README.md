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
|DNNet|ResNet50|Magface| 83.8%| % | 81.2% |
|DNNet|ResNet50|Con+Mag|81.7%|%| 78.7%|
