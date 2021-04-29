# `SkewGP`
### Skew Gaussian Processes:  a unified framework for closed-form nonparametric regression, classification, preference and mixed problems 


![cover](https://github.com/benavoli/SkewGP/blob/main/image.png)

Gaussian Processes (GPs) are powerful nonparametric distributions over functions. For real-valued
outputs, we can combine the GP prior with a Gaussian likelihood and perform exact posterior inference in closed form. However, in other cases, such as classification, preference learning, ordinal regression and mixed problems, the likelihood is no longer conjugate to the GP prior, and exact
inference is known to be intractable.

We showed that is actually possible to derive closed-form expression for the posterior process in all the above cases (not only for regression), and that the posterior process is a Skew Gaussian Process (SkewGP). SkewGPs are more general and more flexible nonparametric distributions than GPs, as SkewGPs may also represent asymmetric distributions. Moreover, SkewGPs include GPs as a particular case. By exploiting the closed-form expression for the posterior and
predictive distribution, we can compute inferences for regression, classification, preference and mixed problems with computation complexity of O(n^3) and storage demands of O(n^2) (same as for GP regression).

This  software library provides a unified framework for closed-form nonparametric inference for a large class of
supervised learning problems.

## Usage
For usage, please refer to the tutorial notebooks

## How to cite
If you are using `SkewGP` for your research, consider citing the following papers: 
```
@article{benavoli2020,
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


@inproceedings{benavoli2021pboSkewGP,
  author = {Benavoli, Alessio and Azzimonti, Dario and Piga, Dario},
  title = {{Preferential Bayesian optimisation with Skew Gaussian Processes}},
  booktitle = {{2021 Genetic and Evolutionary Computation Conference Companion (GECCO '21 Companion), July 10--14, 2021, Lille, France}},
  doi = {10.1145/3449726.3463128},
  isbn = {978-1-4503-8351-6/21/07},
  publisher = {ACM},
  address = {New York, NY, USA},
  keywords = {Bayesian Optimisation, Bayesian preferential optimisation, Skew Gaussian Processes},
  year = {2021}
}



@article{benavoli2021unified,
      title={{A unified framework for closed-form nonparametric regression, classification, preference and mixed problems with Skew Gaussian Processes}}, 
      author={Alessio Benavoli and Dario Azzimonti and Dario Piga},
      year={2021},
      eprint={Arxiv 2012.06846},
      url = {https://arxiv.org/abs/2012.06846}
}

```

