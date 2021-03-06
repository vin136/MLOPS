# MLOPS



[project-ideas](https://docs.google.com/document/d/1wyDSJsunIlheSRXiGCVEnrqYU7RPyvAX2mkAwmMKlMk/edit)

[Distribution-Shift](https://arxiv.org/abs/1711.08534) 

[Uncertanity Estimation](https://arxiv.org/abs/1810.11953)

Resources:

[FullStack DL](https://fullstackdeeplearning.com/spring2021/lecture-11/)

[Made With ML](https://madewithml.com/courses/mlops/baselines/)

[Chip-huyen's book](https://huyenchip.com/machine-learning-systems-design/toc.html)

[ML-SYSTEM DESIGN](https://stanford-cs329s.github.io/2021/syllabus.html)
Go through the slides and read the attached research papers.

[ML-INTERVIEW'S BOOK](https://huyenchip.com/ml-interviews-book/)

Some links from this [reddit-thread](https://www.reddit.com/r/MachineLearning/comments/kayg13/discussion_interview_ml_system_design_prep/)

## Interview-prep
[DataScience](https://github.com/adijo/data-science-prep)

## Writing Good Research Code
[GRC](https://goodresearch.dev/index.html)

[reprdl](https://github.com/sscardapane/reprodl2021)

## Project

A Recommender system using the ideas from - [NCF](https://arxiv.org/abs/2005.08129)

## Tools

1. Hydra: Here's a good [tutorial](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b)
2. Pytorch-lightning(For a well structured code-base)
3. Wand(for hyperparmeter tuning and experment tracking)
4. Githooks,testing,styling.(Good practices)

Extra: Not really necessary for every research project.

Data-Versioning: DVC

Containerizaiton: Docker

`Summary`

For mac M1 users to install sklearn.

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

```
Now change the path, use your `user name`
```
echo 'eval $(/opt/homebrew/bin/brew shellenv)' >> /Users/"YOUR USER NAME"/.zprofile
```
```
eval $(/opt/homebrew/bin/brew shellenv)
brew install openblas
export OPENBLAS=$(/opt/homebrew/bin/brew --prefix openblas)
export CFLAGS="-falign-functions=8 ${CFLAGS}"
# ^ no need to add to .zshrc, just doing this once.
pip install scikit-learn # ==0.24.1 if you want
```

**Flags**

Multirun flag

eg:python main.py lr=1e-3,1e-2 wd=1e-4,1e-2 -m

Print config

eg:python main.py --cfg job

Modify

eg:python train.py data.batch_size=4

Create a flag and set it.

eg:python train.py +trainer.fast_dev_run=True

Remove a flag

eg:python train.py ~trainer.gpus

2. Pytorch-lightning+Hydra for configuration.


