# Human-Motion-Analysis-with-Deep-Metric-Learning
pytorch implement of this paper:https://arxiv.org/abs/1807.11176 (ECCV 2018)

Implement by:

Tim Ren, Harrison Huang

### To do:
- [x] MMD-NCA Loss
- [ ] Layer Normalization LSTM
- [x] Self-Attention
- [x] Training
- [ ] Improve Dataloader

Instead of a Bi-direction Layer Normalization LSTM, we use a non-normalizaiton bi-direction GRU version.
And for now, the dataloader may use a large memory of your cpu, if there is any problems, make the parameter: num_MMD_NCA_Groups of "MMD_NCA_Dataset" smaller.

### Dataset:
I clean the dance dataset of https://arxiv.org/abs/1801.07388
The cleaned dataset is provided here:

https://drive.google.com/file/d/17mUfFjPCZFyZaEyM7NwpLEptg3Vo9DuU/view?usp=sharing

The dataset contains 16 classes of dance. It contain 51858 sequence.
The key of the json file is "0","1",.....,"15"
Each key contains:
( _ , 50, 2, 17) pose. 2 is channel, 17 is pose coordinates as coco format.
And each pose is normalized.

<img src="/image/number_of_sequence.png" width="250">

### Usage:
```
cd Human-Motion-Analysis-with-Deep-Metric-Learning
mkdir log
mkdir dataset
```
Download the dataset I provided above, put it in the folder "dataset".
It is suggested to split it by yourself, for the dataset is too large.

Note: if you split the data, you need to change line 247 in train.py.

And run:
``` 
python train.py
``` 

### Result:

![Alt text](/image/visual_result.png)

### Contact:

If have any question, feel free to connect me by email: xrenaa1998@gmail.com

