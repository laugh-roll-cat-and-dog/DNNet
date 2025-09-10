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

| NN | Backbone | Loss | Acc | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet50|Arcface| 85.9%| 84.6% |0.9272 |
|DNNet|ResNet50|Arc+Soft| 88.4%| 86.0% | 0.9296|
|DNNet|ResNet50|Arc+Focal| 86.3%|  84.0% | 0.9136| 
|DNNet|ResNet50|Arc+Focal+Soft| 86.1%|  83.2% | 0.9145|
|DNNet|ResNet50|Magface| 86.0%|  84.0%|0.9134 |
|DNNet|ResNet50|Mag+Soft| 85.1%| 81.7% | 0.9033|
|DNNet|ResNet50|Mag+Focal| 86.2%| 84.0%| 0.9208|
|DNNet|ResNet50|Mag+Focal+Soft| 86.4%| 82.9% | 0.9124|

epoch = 100, dataset 7530 images, 1500 id, batch size = 32

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet50|Arcface| 87.5%| 85.7% | 0.9268|
|DNNet|ResNet50|Arc+Soft| 86.4%| 85.2% | 0.9223|
|DNNet|ResNet50|Arc+Focal| 85.9%|  83.6% |0.9124 |
|DNNet|ResNet50|Arc+Focal+Soft| 86.2%|  83.1% | 0.9119|
|DNNet|ResNet50|Magface| 86.3%|  83.5%| 0.9116|
|DNNet|ResNet50|Mag+Soft| 86.5%| 83.6% | 0.9055|
|DNNet|ResNet50|Mag+Focal| 86.1%| 83.6%| 0.9076|
|DNNet|ResNet50|Mag+Focal+Soft| 85.4%|  83.0% | 0.8955|

epoch = 100, dataset 7530 images, 1500 id, batch size = 64

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet50|Arcface| 87.1%| 84.2% |0.9127 |
|DNNet|ResNet50|Arc+Focal| 85.5%|  84.2% | 0.9107|
|DNNet|ResNet50|Magface| 85.2%|  81.7%| 0.8930|
|DNNet|ResNet50|Mag+Focal| 86.7%| 82.5%| 0.8995|

epoch = 100, dataset 7530 images, 1500 id, batch size = 128

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet50|Arcface| 85.2%| 82.9% |0.8966 |
|DNNet|ResNet50|Arc+Focal| 85.7%|  82.4%| 0.9003|
|DNNet|ResNet50|Magface| 84.0%|  81.3%| 0.8854|
|DNNet|ResNet50|Mag+Focal| 84.5%| 81.6%| 0.8907|

# Test lambda_g in Magface

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g = 20

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet50|Magface| 86.0%|  84.0%| |
|DNNet|ResNet50|Mag+Soft| 84.9%| 83.4% | |
|DNNet|ResNet50|Mag+Focal| 86.2%| 84.0%| |
|DNNet|ResNet50|Mag+Focal+Soft| 86.3%|  82.0% | |

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g = 15

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet50|Magface| 86.5%|  83.8%| 0.9084|
|DNNet|ResNet50|Mag+Soft| 85.8%| 81.2% | 0.8916|
|DNNet|ResNet50|Mag+Focal| 85.3%| 82.7%| 0.9106|
|DNNet|ResNet50|Mag+Focal+Soft| 85.2%|  81.4% |0.8928 |

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g = 10

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet50|Magface| 85.3%|  83.3%| 0.9139|
|DNNet|ResNet50|Mag+Soft| 85.2%| 81.8% | 0.8938|
|DNNet|ResNet50|Mag+Focal| 85.3%| 82.5%| 0.9020|
|DNNet|ResNet50|Mag+Focal+Soft| 85.1%| 81.7% | 0.8983|

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g = 5

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--| 
|DNNet|ResNet50|Magface| 86.6%|  84.6%| 0.9167|
|DNNet|ResNet50|Mag+Soft| 85.1%| 81.8% |0.8899 |
|DNNet|ResNet50|Mag+Focal| 86.5%| 85.1%|0.9262 |
|DNNet|ResNet50|Mag+Focal+Soft| 84.9%| 81.1% | 0.8846|

epoch = 100, dataset 7530 images, 1500 id, batch size = 16, lambda_g = 5

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet50|Magface| 86.6%|  85.1%| 0.9146|
|DNNet|ResNet50|Mag+Soft|86.2%| 82.4% | 0.9030|
|DNNet|ResNet50|Mag+Focal| 85.6%| 82.7%| 0.9063|
|DNNet|ResNet50|Mag+Focal+Soft| 84.5%| 82.5% | 0.9041|

# Test ResNet-152

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g(magface) = 5

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|--|
|DNNet|ResNet152|Arcface| 84.8%|  82.5%| 0.8995|
|DNNet|ResNet152|Arc+Soft| 84.6%| 82.9% | 0.9105|
|DNNet|ResNet152|Arc+Focal|85.4 %| 82.2% |0.9033 |
|DNNet|ResNet152|Arc+Focal+Soft| 86.6%| 83.4% |0.9117|
|DNNet|ResNet152|Magface| 85.4%|  82.8%| 0.9091|
|DNNet|ResNet152|Mag+Soft| 85.3%| 81.5% | 0.8983|
|DNNet|ResNet152|Mag+Focal| 84.5%| 82.4%|0.8893 |
|DNNet|ResNet152|Mag+Focal+Soft| 84.1%| 80.9%|0.8862 |

epoch = 200, dataset 7530 images, 1500 id, batch size = 32, lambda_g(magface) = 5

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|---|
|DNNet|ResNet152|Arcface| 85.0%|  83.3%| 0.9025|
|DNNet|ResNet152|Arc+Soft| 86.0%| 83.4% | 0.9129|
|DNNet|ResNet152|Arc+Focal|84.9%| 83.5% | 0.8900|
|DNNet|ResNet152|Arc+Focal+Soft| 83.8%|81.5% | 0.8993|
|DNNet|ResNet152|Magface| 85.1%|  82.3%| 0.8967|
|DNNet|ResNet152|Mag+Soft| 84.0%| 80.3% |  0.8928|
|DNNet|ResNet152|Mag+Focal|84.6%| 81.5%| 0.8950|
|DNNet|ResNet152|Mag+Focal+Soft| 83.1%| 80.9% |0.8849 |

# ResNet-152 from BYOL Coco 300 epoch

epoch = 100, dataset 7530 images, 1500 id, batch size = 16, lambda_g(magface) = 5

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|---|
|DNNet|ResNet152|Arcface|88.9%| 88.1%| 0.9318|
|DNNet|ResNet152|Arc+Soft| 90.7%| 88.9% | 0.9465|
|DNNet|ResNet152|Arc+Focal|89.4%| 87.4% | 0.9396|
|DNNet|ResNet152|Arc+Focal+Soft| 90.0%|88.6% |0.9383|
|DNNet|ResNet152|Magface| 89.0%| 88.0%| 0.9389|
|DNNet|ResNet152|Mag+Soft| 88.2%| 87.0% | 0.9325 |
|DNNet|ResNet152|Mag+Focal|88.8%| 88.3%| 0.9442|
|DNNet|ResNet152|Mag+Focal+Soft|90.1%| 87.9% | 0.9465|

epoch = 100, dataset 7530 images, 1500 id, batch size = 32, lambda_g(magface) = 5

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|---|
|DNNet|ResNet152|Arcface| 88.8%|  86.8%| 0.9321|
|DNNet|ResNet152|Arc+Soft| 88.8%| 87.4% | 0.9361|
|DNNet|ResNet152|Arc+Focal|88.6%| 85.5% | 0.9304|
|DNNet|ResNet152|Arc+Focal+Soft| 90.2%| 86.7% | 0.9384|
|DNNet|ResNet152|Magface| 87.9%| 85.6%| 0.9240|
|DNNet|ResNet152|Mag+Soft| 88.3%| 85.5% |0.9266 |
|DNNet|ResNet152|Mag+Focal|88.7%| 86.2% |  0.9294|
|DNNet|ResNet152|Mag+Focal+Soft|87.9%| 85.0% | 0.9140|

# ResNet-50 from BYOL Coco 300 epoch

epoch = 100, dataset 7530 images, 1500 id, batch size = 16, lambda_g(magface) = 5

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|---|
|DNNet|ResNet50|Arcface|%| %| |
|DNNet|ResNet50|Arc+Soft| %| % | |
|DNNet|ResNet50|Arc+Focal|%| % | |
|DNNet|ResNet50|Arc+Focal+Soft| %|% ||
|DNNet|ResNet50|Magface| %| %| |
|DNNet|ResNet50|Mag+Soft| %| % |  |
|DNNet|ResNet50|Mag+Focal|%| %| |
|DNNet|ResNet50|Mag+Focal+Soft|%| % | |


## template

epoch = 200, dataset 7530 images, 1500 id, batch size = 16, lambda_g(magface) = 5

| NN | Backbone | Loss | Acc  | Err Acc| AUC |
|----|------|-----|-----|----|---|
|DNNet|ResNet152|Arcface| %|  %||
|DNNet|ResNet152|Arc+Soft| %| % | |
|DNNet|ResNet152|Arc+Focal|%| % | |
|DNNet|ResNet152|Arc+Focal+Soft| %|% ||
|DNNet|ResNet152|Magface| %| %| |
|DNNet|ResNet152|Mag+Soft| %| % |  |
|DNNet|ResNet152|Mag+Focal|%| %| |
|DNNet|ResNet152|Mag+Focal+Soft|%| % | |