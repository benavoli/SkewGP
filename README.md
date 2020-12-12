# `SkewGP`
### Skew Gaussian Processes:  a unified framework for closed-form nonparametric regression, classification, preference and mixed problems 


![cover](https://github.com/benavoli/SkewGP/blob/main/image.png)

Gaussian Processes (GPs) are powerful nonparametric distributions over functions. For real-valued
outputs, we can combine the GP prior with a Gaussian likelihood and perform exact posterior inference in closed form. However, in other cases, such as classification, preference learning, ordinal regression and mixed problems, the likelihood is no longer conjugate to the GP prior, and exact
inference is known to be intractable.
We showed that is actually possible to derive closed-form expression for the posterior process in all the above cases (not only for regression), and that the posterior process is a Skew Gaussian Process (SkewGP). SkewGPs are more general and more flexible nonparametric distributions than GPs, as SkewGPs may also represent asymmetric distributions. Moreover, SkewGPs include GPs as a particular case. By exploiting the closed-form expression for the posterior and
predictive distribution, we can compute inferences for regression, classification, preference and mixed problems with computation complexity of $O(n^3)$ and storage demands of $O(n^2)$ (same as for GP regression).
This  allows us to provide a unified framework for nonparametric inference for a large class of
likelihoods and, consequently, supervised learning problems.

## Usage
For usage, please refer to the tutorial notebooks

## How to cite
If you are using `LinConGauss` for your research, consider citing both the papers: 
```
@article{benavoli2020a,
title = {Skew Gaussian Processes for Classification},
author = {Alessio Benavoli and Dario Azzimonti and Dario Piga},
url = {https://arxiv.org/abs/2005.12987},
doi = {10.1007/s10994-020-05906-3},
year = {2020},
date = {2020-09-04},
journal = {Machine Learning},
volume = {109},
pages = {1877–1902}
}
```

