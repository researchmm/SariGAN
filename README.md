# SariGAN
Code for NeurIPS 2020 spotlight paper ["Learning Semantic-aware Normalization for Generative Adversarial Networks"](https://papers.nips.cc/paper/2020/file/f885a14eaf260d7d9f93c750e1174228-Paper.pdf)

This code is mainly based on https://github.com/rosinality/stylegan2-pytorch.git 

Please refer to the above link to prepare data and install dependencies.


To train SariGAN, please run:

bash train.sh

To test the model in the term of fid, please run:

bash fid.sh


Pretrained model can be found in https://drive.google.com/file/d/1eT2QEK-0ZNQFj-daL3ga6QOoXqBLe8oR/view?usp=sharing

Citation

@article{zheng2020learning,
  title={Learning Semantic-aware Normalization for Generative Adversarial Networks},
  author={Zheng, Heliang and Fu, Jianlong and Zeng, Yanhong and Luo, Jiebo and Zha, Zheng-Jun},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={21853--21864},
  year={2020}
}
