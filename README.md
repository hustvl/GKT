<div align="center">
<h1> Geometry-guided Kernel Transformer </h1>
<span><font size="5", > Efficient and Robust 2D-to-BEV Representation Learning via Geometry-guided Kernel Transformer
 </font></span>
</br>
Shaoyu Chen*, Tianheng Cheng*, <a href="https://xinggangw.info/">Xinggang Wang</a><sup><span>&#8224;</span></sup>, Wenming Meng, <a href="https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN">Qian Zhang</a>, <a href="http://eic.hust.edu.cn/professor/liuwenyu/"> Wenyu Liu</a>

(<span>*</span>: equal contribution, <span>&#8224;</span>: corresponding author)
<br>
<div><a href="https://arxiv.org/pdf/2206.04584.pdf">[arXiv Preprint]</a></div> 

</div>

## News

* `June 9, 2022`: We've released the tech report for Geometry-guided Kernel Transformer (GKT). This work is still in progress and code/models are comming sonn. Please stay tuned! ☕️

## Introduction

![Framework](./assets/GKT-main.png)

We present a novel and efficient **2D-to-BEV** transformation, Geometry-guided Kernel Transformer (GKT)

* GKT leverages geometric priors to guide the transformers to focus on discriminative regions for generating BEV representation with surrouding-view image features.
* GKT is based on kernel-wise attention and much efficient, especially with LUT indexing.
* GKT is robust to the deviation of cameras, making the 2D-to-BEV transformation more stable and reliable.

## Models

coming soon.

## Usage

coming soon.


## License

GKT is released under the [MIT Licence](LICENSE).

## Citation

If you find GKT is useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.

```bibtex
@article{GeokernelTransformer,
  title={Efficient and Robust 2D-to-BEV Representation Learning via Geometry-guided Kernel Transformer},
  author={Chen, Shaoyu and Cheng, Tianheng and Wang, Xinggang and Meng, Wenming and Zhang, Qian and Liu, Wenyu},
  journal={arXiv preprint arXiv:2206.04584},
  year={2022}
}
```
