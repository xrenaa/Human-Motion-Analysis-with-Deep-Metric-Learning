# Human-Motion-Analysis-with-Deep-Metric-Learning
pytorch implement of this paper:https://arxiv.org/abs/1807.11176 (ECCV 2018)

### To do:
- [x] MMD-NCA Loss
- [ ] Layer Normalization LSTM
- [x] Self-Attention
- [x] Training

Instead of a Bi-direction Layer Normalization LSTM, we use a non-normalizaiton bi-direction GRU version.

### Dataset:
I clean the dance dataset of https://arxiv.org/abs/1801.07388
The cleaned dataset is provided here:

The dataset contains 16 classes of dance. 


### Result:

![Alt text](/image/visual_result.png)

