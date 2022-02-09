# Kac independence measure 
Kac independence measure (KacIM) is bivariate statistical independence measure, 
which can detect arbitrary statistical dependence between two random vectors (similar to mutual information, Hilbert-Schmidt independence criterion (HSIC), etc.). The idea of KacIM is to maximimize lenght of difference 
between joint and product marginal characteristic functions (two complex random variables):

![Alt text](./kac_im.png?raw=true "KacIM")


This repository includes basic implementation of KacIM, and one very basic demonstration, which show that KacIM works for high-dimensional data (e.g. 512-dimensional input, 4-dimensional output or similar).


Article/preprint is currently being prepared: ![Article](https://github.com/povidanius/kac_independence_measure/tree/main/art/main.pdf?raw=false "Article")

<object data="https://github.com/povidanius/kac_independence_measure/tree/main/art/main.pdf?raw=false" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/povidanius/kac_independence_measure/tree/main/art/main.pdf?raw=false">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/povidanius/kac_independence_measure/tree/main/art/main.pdf?raw=false">Download PDF</a>.</p>
    </embed>
</object>
This how KacIM evaluations during gradient optimization looks like for independent data (blue), dependent data with additive (orange) and multiplicative noise (green) (500 iterations):

![Alt text](./independent_dependent.png?raw=true "Title")

In independent case the estimator does not converge, meanwhile in dependent cases it does.




