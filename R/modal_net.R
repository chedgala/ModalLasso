#' Penalized parametric modal regression for positive responses
#'
#' Fits the modal regression model \eqn{\log M_i = \gamma_0 + x_i^\top\gamma},
#' where \eqn{M_i} is the conditional mode of a positive response, by
#' maximizing the penalized log-likelihood
#' \deqn{\ell_p(\gamma,\phi) = \ell(\gamma,\phi)/n - \lambda\,\Psi(\gamma),}
#' with \eqn{\Psi} the Lasso, elastic-net or adaptive-Lasso penalty on the
#' slopes (the intercept is never penalized). The Gamma, Weibull and Lognormal
#' families are available, each written in terms of its mode through the
#' reparameterization of Galarza and Lachos (2026).
#'
#' @details
#' The estimator is computed by a \emph{proximal Newton} algorithm, equivalently
#' an iteratively reweighted penalized least-squares (IRPLS) scheme. At each
#' iteration the modal log-likelihood is replaced by its second-order expansion
#' in the linear predictor \eqn{\eta_i}, which yields a pseudo-response
#' \eqn{z_i} and a working weight \eqn{w_i} in closed form:
#' \itemize{
#'   \item Gamma: \eqn{w_i = y_i/(\phi M_i)}, \eqn{z_i = \eta_i + 1 - (1+\phi)M_i/y_i};
#'   \item Weibull: \eqn{w_i = k^2 (y_i/\lambda_i)^k}, \eqn{z_i = \eta_i + \{1 - (\lambda_i/y_i)^k\}/k};
#'   \item Lognormal: \eqn{w_i = 1/\phi}, \eqn{z_i = \log y_i - \phi} (the surrogate is exact).
#' }
#' The update of \eqn{\gamma} is then the weighted penalized least-squares
#' problem \eqn{\min_\gamma \tfrac12\sum_i w_i (z_i - x_i^\top\gamma)^2 + \lambda\Psi(\gamma)},
#' solved exactly by the cyclical coordinate descent of \pkg{glmnet} with
#' observation weights, so that variable selection is performed by the
#' optimizer itself (exact zeros, no thresholding). An optional backtracking
#' line search keeps the penalized objective monotone. The dispersion
#' \eqn{\phi} is then updated by maximizing the profile log-likelihood (closed
#' form for the Lognormal family, one-dimensional optimization otherwise).
#'
#' The regularization parameter is chosen by path following: a decreasing grid
#' \eqn{\lambda_1 > \cdots > \lambda_L} is built once from the initial
#' pseudo-response, the algorithm is run to convergence at each grid point with
#' a warm start at the previous solution, and the extended BIC
#' \deqn{\mathrm{EBIC}(\lambda) = -2\ell + \mathrm{df}\log n + 2\gamma_{\mathrm{EBIC}}\,\mathrm{df}\log p,
#' \qquad \mathrm{df} = |\hat S_\lambda| + 2,}
#' is evaluated across the converged fits. The initialization uses a lightly
#' penalized Gaussian Lasso of \eqn{\log y} on \eqn{X}, which stays well
#' defined when \eqn{p > n}.
#'
#' The penalty \eqn{\lambda} is expressed on the scale used by \pkg{glmnet},
#' whose weights are rescaled to sum to \eqn{n}; as noted in the paper, this is
#' what makes the Lognormal fit at penalty \eqn{\lambda} coincide with the
#' Gaussian Lasso of \eqn{\log y} at the \emph{same} \eqn{\lambda} (slopes
#' identical, intercepts differing by exactly \eqn{\phi}).
#'
#' @param y numeric vector of positive responses.
#' @param X numeric matrix (or data frame) of covariates, \eqn{n \times p},
#'   without an intercept column.
#' @param family one of \code{"gamma"}, \code{"weibull"}, \code{"lognormal"}.
#' @param lambda optional user-supplied decreasing sequence of penalty values.
#'   If \code{NULL} (default) a grid of length \code{nlambda} is built.
#' @param nlambda length of the automatic grid.
#' @param lambda.min.ratio smallest grid value as a fraction of the largest;
#'   default \code{0.01} when \eqn{n > p} and \code{0.05} otherwise.
#' @param alpha elastic-net mixing parameter passed to \pkg{glmnet}:
#'   \code{alpha = 1} (default) is the Lasso, \code{0 < alpha < 1} the elastic
#'   net and \code{alpha = 0} ridge regression. Ridge produces no exact zeros,
#'   so \code{selected} contains every covariate and the EBIC, which counts
#'   active coefficients, is not a selection criterion there; compare ridge
#'   fits by prediction instead.
#' @param penalty.factor optional vector of length \eqn{p} of relative penalty
#'   weights (adaptive Lasso); zeros leave a variable unpenalized.
#' @param gamma_ebic the EBIC constant \eqn{\gamma_{\mathrm{EBIC}} \in [0,1]};
#'   \code{0} gives the ordinary BIC.
#' @param dfmax the path stops once more than \code{dfmax} covariates are
#'   active; default \code{min(p, 0.8 n)}. Near saturation (\eqn{p \ge n}) the
#'   Gamma and Weibull likelihoods are unbounded as \eqn{\phi \to 0}, so the
#'   dense end of the path carries no information.
#' @param line_search logical; use the backtracking safeguard
#'   \eqn{\alpha^{(k)} \in \{1, 1/2, 1/4, \dots\}} so that the penalized
#'   objective never increases.
#' @param standardize passed to \pkg{glmnet}; standardize the columns of
#'   \code{X} internally before penalizing.
#' @param tol relative tolerance on the penalized objective.
#' @param maxit maximum number of proximal Newton iterations per grid point.
#' @param verbose print progress along the path.
#'
#' @return An object of class \code{"ModalNet"}, a list with components
#' \item{gamma}{estimated coefficients (intercept first) at the EBIC-selected
#'   \eqn{\lambda}.}
#' \item{phi}{estimated dispersion at the selected \eqn{\lambda}.}
#' \item{selected}{indices of the selected (nonzero) covariates.}
#' \item{lambda}{selected penalty.}
#' \item{loglik, ebic, df}{values at the selected penalty.}
#' \item{path}{list with the whole path: \code{lambda}, \code{coef}
#'   (matrix \eqn{(p+1)\times L}), \code{phi}, \code{loglik}, \code{df},
#'   \code{ebic}, \code{iter}, \code{converged}.}
#' \item{fitted.mode}{fitted conditional modes \eqn{\hat M_i}.}
#' \item{mean_const}{the constant \eqn{c(\hat\phi)} with
#'   \eqn{E[Y\mid x] = c(\phi) M}; the conditional mean has the same slopes
#'   and intercept \eqn{\hat\gamma_0 + \log c(\hat\phi)}.}
#' \item{boundary}{\code{TRUE} when the dispersion estimate sits at the limit
#'   of its range (\eqn{10^{-4} \le \phi \le 10^{4}}), which for the Gamma and
#'   Weibull families means a fitted density close to the exponential limit
#'   with mode near zero; a warning is issued.}
#' plus \code{family}, \code{n}, \code{p}, \code{gamma_ebic}, \code{call}.
#'
#' @references
#' Galarza, C. E. and Lachos, V. H. (2026). A proximal Newton algorithm for
#' penalized modal regression with high-dimensional positive responses.
#'
#' Galarza, C. E. and Lachos, V. H. (2026). Parametric modal regression for
#' right-censored positive responses. arXiv:2603.07099.
#'
#' Friedman, J., Hastie, T. and Tibshirani, R. (2010). Regularization paths
#' for generalized linear models via coordinate descent. \emph{Journal of
#' Statistical Software}, 33(1), 1--22.
#'
#' Chen, J. and Chen, Z. (2008). Extended Bayesian information criteria for
#' model selection with large model spaces. \emph{Biometrika}, 95, 759--771.
#'
#' @examples
#' set.seed(1)
#' n <- 200; p <- 30
#' X <- matrix(rnorm(n * p), n, p)
#' gamma_true <- c(1, 0.6, -0.5, 0.4, rep(0, p - 3))
#' dat <- generate_modal_data(X, gamma_true, phi = 0.3, family = "gamma")
#' fit <- modal_net(dat$y, X, family = "gamma")
#' fit
#' summary(fit)
#' plot(fit)
#'
#' @export
modal_net <- function(y, X, family = c("gamma", "weibull", "lognormal"),
                        lambda = NULL, nlambda = 50, lambda.min.ratio = NULL,
                        alpha = 1, penalty.factor = NULL, gamma_ebic = 0.5,
                        dfmax = NULL, line_search = TRUE, standardize = TRUE,
                        tol = 1e-6, maxit = 200, verbose = FALSE) {
  cl <- match.call()
  family <- match.arg(family)
  fam <- .get_family(family)

  y <- as.numeric(y)
  if (any(!is.finite(y)) || any(y <= 0)) stop("'y' must be a finite, strictly positive vector")
  X <- as.matrix(X)
  if (!is.numeric(X)) stop("'X' must be a numeric matrix")
  if (nrow(X) != length(y)) stop("nrow(X) must equal length(y)")
  if (any(!is.finite(X))) stop("'X' contains missing or infinite values")
  n <- length(y); p <- ncol(X)
  if (is.null(colnames(X))) colnames(X) <- paste0("x", seq_len(p))
  if (is.null(penalty.factor)) penalty.factor <- rep(1, p)
  if (is.null(lambda.min.ratio)) {
    lambda.min.ratio <- if (n > p) 0.01 else 0.05
    if (alpha == 0) lambda.min.ratio <- 1e-4   # glmnet inflates lambda_max for ridge
  }
  if (is.null(dfmax)) dfmax <- min(p, max(5, floor(0.8 * n)))

  ## ---- objective pieces --------------------------------------------------
  ## lambda is on the scale used by glmnet: the inner subproblem is
  ##   (1/(2n)) sum_i w'_i (z_i - eta_i)^2 + lambda Psi(gamma),  w' = w * n / sum(w),
  ## which equals -ell/sum(w) + lambda Psi(gamma) up to the surrogate error.
  ## The merit function below uses the same normalization S = sum(w) of the
  ## current iteration, so that the line search compares like with like.
  ## glmnet penalizes the coefficients of the (weighted-)standardized columns,
  ## so the merit function must weigh |gamma_j| by the column scale sd_j used
  ## in the current subproblem; otherwise the line search rejects valid steps.
  pen <- function(g, sdj) {
    b <- g[-1]
    sum(penalty.factor * (alpha * sdj * abs(b) + (1 - alpha) * sdj^2 * b^2 / 2))
  }
  objective <- function(g, phi, lam, S, sdj)
    -fam$loglik(y, .clamp_eta(cbind(1, X) %*% g), phi) / S + lam * pen(g, sdj)
  col_scale <- function(w) {
    if (!standardize) return(rep(1, p))
    wn <- as.numeric(w) / sum(w)
    xbar <- colSums(wn * X)
    sqrt(colSums(wn * sweep(X, 2, xbar)^2))
  }

  ## ---- initialization: lightly penalized Gaussian Lasso of log(y) ----------
  g0 <- glmnet::glmnet(X, log(y), family = "gaussian", alpha = alpha,
                       penalty.factor = penalty.factor, standardize = standardize)
  gam <- as.numeric(stats::coef(g0, s = 0.1 * max(g0$lambda)))
  eta <- .clamp_eta(cbind(1, X) %*% gam)
  phi <- max(stats::var(log(y) - eta), 0.05)
  ## calibrate intercept and dispersion jointly on the modal scale (the Gaussian
  ## fit locates the mean of log y, not the mode of y)
  up <- .update_intercept_phi(fam, y, as.numeric(X %*% gam[-1]), gam[1], phi)
  gam[1] <- up$gamma0; phi <- up$phi
  eta <- .clamp_eta(cbind(1, X) %*% gam)

  ## ---- grid of lambda from the initial pseudo-response ---------------------
  zw <- fam$zw(y, eta, phi)
  if (is.null(lambda)) {
    g1 <- glmnet::glmnet(X, zw$z, weights = zw$w, family = "gaussian", alpha = alpha,
                         penalty.factor = penalty.factor, standardize = standardize,
                         nlambda = nlambda, lambda.min.ratio = lambda.min.ratio)
    lambda <- g1$lambda
  } else {
    lambda <- sort(as.numeric(lambda), decreasing = TRUE)
  }
  L <- length(lambda)

  ## ---- path following with warm starts -------------------------------------
  coefs <- matrix(NA_real_, p + 1, L, dimnames = list(c("(Intercept)", colnames(X)), NULL))
  phis <- logliks <- dfs <- ebics <- iters <- numeric(L)
  convs <- logical(L)

  for (l in seq_len(L)) {
    lam <- lambda[l]
    converged <- FALSE
    for (it in seq_len(maxit)) {
      ## 1. quadratic surrogate at the current iterate
      zw <- fam$zw(y, eta, phi)
      w <- zw$w; z <- zw$z; S <- sum(w); sdj <- col_scale(w)
      obj_old <- objective(gam, phi, lam, S, sdj)
      ## 2. penalized weighted least squares by coordinate descent (glmnet)
      g_fit <- glmnet::glmnet(X, z, weights = w, family = "gaussian", alpha = alpha,
                              penalty.factor = penalty.factor, standardize = standardize,
                              lambda = lam)
      g_new <- as.numeric(stats::coef(g_fit))
      ## 3. backtracking line search on the penalized objective
      step <- 1
      if (line_search) {
        repeat {
          g_try <- gam + step * (g_new - gam)
          if (objective(g_try, phi, lam, S, sdj) <= obj_old + 1e-12 || step < 1 / 64) break
          step <- step / 2
        }
        g_new <- gam + step * (g_new - gam)
      }
      gam_prev <- gam; phi_prev <- phi
      gam <- g_new
      ## 4. dispersion update: profile likelihood over phi, jointly with the
      ##    unpenalized intercept (closed form for the Lognormal family)
      up <- .update_intercept_phi(fam, y, as.numeric(X %*% gam[-1]), gam[1], phi)
      gam[1] <- up$gamma0; phi <- up$phi
      eta <- .clamp_eta(cbind(1, X) %*% gam)
      obj_new <- objective(gam, phi, lam, S, sdj)
      if (abs(obj_new - obj_old) < tol * (1 + abs(obj_old)) ||
          (max(abs(gam - gam_prev)) < 1e-6 && abs(phi - phi_prev) < 1e-6 * (1 + phi))) {
        converged <- TRUE; break
      }
    }
    ll <- fam$loglik(y, eta, phi)
    df <- sum(gam[-1] != 0) + 2
    coefs[, l] <- gam; phis[l] <- phi; logliks[l] <- ll; dfs[l] <- df; iters[l] <- it; convs[l] <- converged
    ebics[l] <- -2 * ll + df * log(n) + 2 * gamma_ebic * df * log(p)
    if (verbose) cat(sprintf("lambda[%2d] = %8.5f  df = %3d  loglik = %10.3f  EBIC = %10.3f  it = %3d\n",
                             l, lam, df - 2, ll, ebics[l], it))
    if (alpha > 0) {                  # sparse penalties only: ridge never produces zeros
      if (df - 2 > dfmax) {           # stop the path before (near-)saturation
        L <- max(l - 1, 1)            # and discard the fit that crossed dfmax
        break
      }
      if (df - 2 == dfmax) { L <- l; break }
    }
  }
  keep <- seq_len(L)
  lambda <- lambda[keep]; coefs <- coefs[, keep, drop = FALSE]
  phis <- phis[keep]; logliks <- logliks[keep]; dfs <- dfs[keep]; ebics <- ebics[keep]
  iters <- iters[keep]; convs <- convs[keep]

  best <- which.min(ebics)
  boundary <- phis[best] <= .phi_bounds[1] * 1.01 || phis[best] >= .phi_bounds[2] * 0.99
  if (boundary && family != "lognormal")
    warning("the dispersion estimate is at the boundary of its range: the fitted ",
            family, " density is close to the exponential limit (mode near zero), ",
            "so the modal parameterization is not well supported by these data")
  gam_b <- coefs[, best]
  eta_b <- .clamp_eta(cbind(1, X) %*% gam_b)
  out <- list(
    gamma = gam_b, phi = phis[best], selected = which(gam_b[-1] != 0),
    lambda = lambda[best], loglik = logliks[best], ebic = ebics[best], df = dfs[best],
    path = list(lambda = lambda, coef = coefs, phi = phis, loglik = logliks, df = dfs,
                ebic = ebics, iter = iters, converged = convs),
    fitted.mode = as.numeric(exp(eta_b)),
    mean_const = fam$mean_const(phis[best]),
    family = family, n = n, p = p, gamma_ebic = gamma_ebic, alpha = alpha,
    boundary = boundary, dfmax = dfmax,
    xnames = colnames(X), y = y, X = X, call = cl)
  class(out) <- "ModalNet"
  out
}

#' Simulate data from a modal regression model
#'
#' Generates positive responses whose conditional mode is
#' \eqn{M_i = \exp(\gamma_0 + x_i^\top\gamma)} under the Gamma, Weibull or
#' Lognormal mode reparameterization used by \code{\link{modal_net}}.
#'
#' @param X covariate matrix without intercept.
#' @param gamma coefficient vector of length \code{ncol(X) + 1} (intercept first).
#' @param phi dispersion parameter.
#' @param family one of \code{"gamma"}, \code{"weibull"}, \code{"lognormal"}.
#' @return A list with \code{y}, \code{mode} and \code{mean}.
#' @examples
#' X <- matrix(rnorm(100 * 5), 100, 5)
#' d <- generate_modal_data(X, c(1, 0.5, -0.5, 0, 0, 0), phi = 0.3, family = "weibull")
#' str(d)
#' @export
generate_modal_data <- function(X, gamma, phi, family = c("gamma", "weibull", "lognormal")) {
  family <- match.arg(family)
  X <- as.matrix(X)
  stopifnot(length(gamma) == ncol(X) + 1, phi > 0)
  M <- as.numeric(exp(cbind(1, X) %*% gamma))
  n <- length(M)
  y <- switch(family,
    gamma = { alpha <- 1 / phi + 1; stats::rgamma(n, shape = alpha, scale = phi * M) },
    weibull = { k <- phi + 1.01; stats::rweibull(n, shape = k, scale = M * (k / (k - 1))^(1 / k)) },
    lognormal = stats::rlnorm(n, meanlog = log(M) + phi, sdlog = sqrt(phi)))
  list(y = y, mode = M, mean = .get_family(family)$mean_const(phi) * M)
}


#' Alias of \code{modal_net} kept for compatibility
#'
#' \code{modal_lasso()} was the main function of the package when it was called
#' \pkg{ModalLasso}. It calls \code{\link{modal_net}} with the same arguments,
#' so existing code keeps working. New code should use \code{modal_net()}.
#'
#' @param ... arguments passed to \code{\link{modal_net}}.
#' @return The same object as \code{\link{modal_net}}.
#' @export
modal_lasso <- function(...) modal_net(...)
