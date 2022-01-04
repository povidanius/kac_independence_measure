# Kac independence measure 
Kac independence measure (KacIM) is bivariate statistical independence measure, 
which can detect arbitrary statistical dependence between two random vectors (similar to mutual information, Hilbert-Schmidt independence criterion (HSIC), etc.). The idea of KacIM is to maximimize lenght of difference 
between joint and product marginal characteristic functions (two complex random variables):

![Alt text](./kac_im.png?raw=true "KacIM")


This repository includes basic implementation of KacIM, and one very basic demonstration, which show that KacIM works for high-dimensional data (e.g. 1024-dimensional input, 32-dimensional output or similar).


Article/preprint is currently being prepared: ![Article](https://github.com/povidanius/kac_independence_measure/tree/main/art/main.pdf?raw=false "Article")


This how KacIM evaluations during gradient optimization looks like for independent data (2000 iterations):

![Alt text](./independent.png?raw=true "Title")

The pattern in this case is continuous oscillation very slightly above (near) zero.

And here how it looks like for dependent data:

![Alt text](./dependent.png?raw=true "Title")

In this case the pattern is maximizability.



