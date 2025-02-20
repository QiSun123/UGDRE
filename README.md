# UGDRE
Code for ACL 2023 paper [Uncertainty Guided Label Denoising for Document-level Distant Relation Extraction.](https://aclanthology.org/2023.acl-long.889/)
## Installation
```
  conda env create -f environment.yml
  conda activate UGDRE
```
## Dataset
We perform experiments on [DocRED](https://github.com/thunlp/DocRED) and [RE-DocRED](https://github.com/tonytan48/re-docred).
## Our Denoised data
For the DocRED dataset, our denoised data can be found at this [link](https://drive.google.com/file/d/1Rk1bNJgZqQkQwtvNGuWzqzqSs_Z_B5TD/view?usp=sharing). 
For the RE-DocRED dataset, our denoised data can be found at this [link](https://drive.google.com/file/d/1yyQyQlAWxKL1FZmoaWWWGjB7_sNOPLvD/view?usp=sharing).
## Training and Evaluation
### Pretrain the DRE model with DS data:
```
  bash scripts/run_pretrain.sh
```
### Fine-tune the DRE model with human-annotated data:
```
  bash scripts/run_finetune.sh
```
### Generate pseudo instances with uncertainty scores
```
  bash scripts/generate_pseudo_uncertainty.sh
```
### Perform a re-label strategy to obtain denoised DS data
```
  python dataset.py
```
Part of the code is adapted from [ATLOP.](https://github.com/wzhouad/ATLOP)
## Citation
```
@article{sun2023uncertainty,
  title={Uncertainty Guided Label Denoising for Document-level Distant Relation Extraction},
  author={Sun, Qi and Huang, Kun and Yang, Xiaocui and Hong, Pengfei and Zhang, Kun and Poria, Soujanya},
  journal={Proceedings of ACL},
  year={2023}
}
```



  



