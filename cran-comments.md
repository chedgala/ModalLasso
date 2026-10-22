## Submission

This is the first submission of ModalNet to CRAN.

The package implements penalized (Lasso, elastic net, ridge and adaptive
Lasso) parametric modal regression for positive responses (Gamma, Weibull and
Lognormal families), using 'glmnet' as the inner solver of a proximal Newton
algorithm.

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Christian E. Galarza <chedgala@espol.edu.ec>'
  New submission

## Test environments

* local: Windows 11 x64, R 4.5.0 (R CMD check --as-cran)
* win-builder: R-devel, R-release and R-oldrelease
* GitHub Actions: ubuntu-latest (R-devel, R-release, R-oldrel-1),
  macos-latest (R-release), windows-latest (R-release)
