# Kac independence measure 
Kac independence measure (KacIM) is bivariate statistical independence measure, 
which can detect arbitrary statistical dependence between two random vectors (similar to mutual information, Hilbert-Schmidt independence criterion (HSIC), distance covariance/correlation, etc.). The idea of KacIM is to maximimize lenght of difference 
between joint and product marginal characteristic functions (two complex random variables):

![Alt text](./kac_im.png?raw=true "KacIM")


This repository includes basic implementation of KacIM, toy-data demonstrations, which show that KacIM works for high-dimensional data (e.g. 512-dimensional input, 4-dimensional output or similar), and feature extraction example, which demonstrates, that KacIM allows to improve classification accuracy on real data.



Article/preprint is currently being prepared: [Article draft](https://github.com/povidanius/kac_independence_measure/tree/main/art/main.pdf?raw=false "Article draft").
In this article we show theoretically how KacIM is related to distance correlation. Also we point out connection with canonical correlation analysis.
From the empirical aspect of our study, we investigate both generated data and real data scenarios. 

This how KacIM evaluations during gradient optimization looks like for independent data (blue), dependent data with additive (orange) and multiplicative noise (green) (500 iterations):

![Alt text](./independent_dependent.png?raw=true "Title")

In independent case the estimator does not converge, meanwhile in dependent cases it does.




