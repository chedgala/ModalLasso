# ModalLasso

**Penalized parametric modal regression for high-dimensional positive responses**

Christian E. Galarza (ESPOL, Ecuador) · Víctor H. Lachos (University of Connecticut, USA)

`ModalLasso` performs variable selection and estimation in parametric modal regression, where the **conditional mode** of a positive response is linked to a linear predictor:

$$\log M_i = \gamma_0 + \mathbf{x}_i^\top\boldsymbol\gamma, \qquad M_i = \arg\max_y f(y \mid \mathbf{x}_i).$$

The Gamma, Weibull and Lognormal families are written in terms of their mode through the analytical reparameterization of Galarza and Lachos (2026), and the penalized log-likelihood

$$\ell_p(\boldsymbol\gamma,\phi) = \ell(\boldsymbol\gamma,\phi)/n - \lambda\,\Psi(\boldsymbol\gamma)$$

is maximized for the Lasso, elastic-net and adaptive-Lasso penalties.

## Algorithm

The estimator is computed with a **proximal Newton** algorithm (equivalently, iteratively reweighted penalized least squares). At each iteration the modal log-likelihood is replaced by its second-order expansion in the linear predictor, which gives a pseudo-response $z_i$ and a working weight $w_i$ in closed form:

| Family | $w_i$ | $z_i$ |
|---|---|---|
| Gamma ($\alpha = 1/\phi + 1$, $\beta_i = \phi M_i$) | $y_i/(\phi M_i)$ | $\eta_i + 1 - (1+\phi)M_i/y_i$ |
| Weibull ($k = \phi + 1.01$, $\lambda_i = M_i\{k/(k-1)\}^{1/k}$) | $k^2 (y_i/\lambda_i)^k$ | $\eta_i + \{1 - (\lambda_i/y_i)^k\}/k$ |
| Lognormal ($\mu_i = \eta_i + \phi$, $\sigma^2 = \phi$) | $1/\phi$ | $\log y_i - \phi$ |

The update of $\boldsymbol\gamma$ is the weighted penalized least-squares problem

$$\min_{\boldsymbol\gamma}\ \tfrac12\sum_i w_i\,(z_i - \mathbf{x}_i^\top\boldsymbol\gamma)^2 + \lambda\,\Psi(\boldsymbol\gamma),$$

solved exactly by the coordinate-descent solver of **glmnet** with observation weights. glmnet never sees the response $y$: it only sees $(z_i, w_i)$, which carry the modal density. The dispersion $\phi$ is updated by maximizing the profile log-likelihood (closed form for the Lognormal family). A backtracking line search keeps the penalized objective monotone, $\lambda$ follows a warm-started decreasing grid, and the final model is chosen by the **extended BIC** (Chen and Chen, 2008) with $\gamma_{\mathrm{EBIC}} = 0.5$ by default.

The penalty $\lambda$ is on glmnet's scale (observation weights rescaled to sum to $n$), exactly as in the paper. The intercept and the dispersion are updated jointly (with the slopes fixed) after every coordinate-descent step, because in the mode parameterization they are strongly coupled.

Two facts proved in the paper and visible in the package output:

* **Lognormal equivalence.** The Lognormal modal Lasso at penalty $\lambda$ *is* the Gaussian Lasso on $\log y$ at the same $\lambda$: identical slopes, intercepts differing by exactly $\phi$.
* **Mode and mean differ only in the intercept.** With a log link and constant dispersion, $E[Y \mid \mathbf{x}] = c(\phi)\,M(\mathbf{x})$ with $c(\phi) = 1+\phi$ (Gamma), $\{k/(k-1)\}^{1/k}\Gamma(1+1/k)$ (Weibull), $e^{3\phi/2}$ (Lognormal). `summary()` reports $c(\hat\phi)$ and the mean-scale intercept.

## Installation

```r
# install.packages("remotes")
remotes::install_github("chedgala/ModalLasso")
```

## Quick start

```r
library(ModalLasso)
set.seed(1)
n <- 200; p <- 50
X <- matrix(rnorm(n * p), n, p)
gamma_true <- c(1, 0.6, -0.5, 0.4, rep(0, p - 3))
dat <- generate_modal_data(X, gamma_true, phi = 0.3, family = "gamma")

fit <- modal_lasso(dat$y, X, family = "gamma")
fit                      # selected variables and coefficients on the log-mode scale
summary(fit)             # exp(coef), phi, EBIC, mode-mean constant
plot(fit)                # EBIC curve and coefficient path
predict(fit, type = "mode")[1:5]
predict(fit, type = "mean")[1:5]
```

### Comparing with the mean-based (log-Gaussian) Lasso

```r
library(glmnet)
cv <- cv.glmnet(X, log(dat$y))
coef(cv, s = "lambda.1se")          # Lasso on log(y): targets the mean of log y
coef(fit)                           # modal Lasso: targets the mode of y
```

For the Lognormal family the two paths coincide (`modal_lasso(..., family = "lognormal")` versus `glmnet(X, log(y))` at the same `lambda`), which is Proposition 1 of the paper; for the Gamma and Weibull families the working weights vary with the data and the fit, and the two selections differ.

## Applications in the paper

* VA lung cancer trial (`survival::veteran`, 128 complete cases, 9 covariates).
* Riboflavin production, $n = 71$, $p = 100$ genes with the largest variance. The subset is bundled as `data(riboflavinV100)` (derived from the `riboflavin` data of the `hdi` package, Bühlmann, Kalisch and Meier, 2014).

```r
library(survival)
v  <- veteran[veteran$status == 1, ]                     # 128 complete cases
Xv <- model.matrix(~ trt + karno + diagtime + age + prior + celltype, v)[, -1]
fit_g <- modal_lasso(v$time, Xv, family = "gamma")
fit_w <- modal_lasso(v$time, Xv, family = "weibull")
colnames(Xv)[fit_g$selected]; colnames(Xv)[fit_w$selected]

data(riboflavinV100)
y <- riboflavinV100$y / mean(riboflavinV100$y)
fit_r <- modal_lasso(y, riboflavinV100$x, family = "gamma")
colnames(riboflavinV100$x)[fit_r$selected]
```

## References

* Galarza, C. E. and Lachos, V. H. (2026). *A proximal Newton algorithm for penalized modal regression with high-dimensional positive responses*. Manuscript.
* Galarza, C. E. and Lachos, V. H. (2026). *Parametric modal regression for right-censored positive responses*. arXiv:2603.07099. R package [ModalCens](https://github.com/chedgala/ModalCens).
* Friedman, J., Hastie, T. and Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *JSS* 33(1).
* Chen, J. and Chen, Z. (2008). Extended Bayesian information criteria for model selection with large model spaces. *Biometrika* 95, 759–771.

## License

GPL-3
