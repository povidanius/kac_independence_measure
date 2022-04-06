# Kac Independence Measure 
Kac Independence Measure (KacIM) is bivariate statistical independence measure, 
which can detect arbitrary statistical dependence between two random vectors (similar to mutual information, Hilbert-Schmidt independence criterion (HSIC), distance covariance/correlation, etc.). The idea of KacIM is to maximimize lenght of difference 
between joint and product marginal characteristic functions (two complex random variables):

![Alt text](./kac_im.png?raw=true "KacIM")


This repository includes basic implementation of KacIM, toy-data demonstrations, which show that KacIM works for high-dimensional data (e.g. 512-dimensional input, 512-dimensional output or similar), and feature extraction example, which demonstrates, that KacIM allows to improve classification accuracy on real data. In generated data scenario we provide empirical analysis of statistical indepndence and non-linear statistical dependene with additive noie. Therein we compare KacIM with distance correlation, and show that KacIM is not affected by curse of dimensionality.



Article/preprint is currently being prepared: [Article draft](https://github.com/povidanius/kac_independence_measure/tree/main/art/main.pdf?raw=false "Article draft").
In this article we identify that KacIM is related to distance correlation in common $L^{p}$-space framework. Also we point out connection with canonical correlation analysis.
From the empirical aspect of our study, we investigate both generated data and real data scenarios. 

Example:
This graph show KacIM evaluations during gradient optimization looks like for independent data (blue), dependent data with additive (orange) and multiplicative noise (green) (500 iterations):

![Alt text](./independent_dependent.png?raw=true "Title")

In independent case the estimator does not converge, meanwhile in dependent cases it does.




