library(ModalNet)

## Lognormal modal fit = Gaussian fit of log(y) (same slopes) for every alpha
set.seed(10)
n <- 120; p <- 8
X <- matrix(rnorm(n * p), n, p)
dat <- generate_modal_data(X, c(1, 0.6, -0.4, rep(0, p - 2)), phi = 0.3, family = "lognormal")
ly <- log(dat$y); sly <- sqrt(mean((ly - mean(ly))^2))
for (a in c(1, 0.5, 0)) {
  lam <- 0.05
  fit <- modal_net(dat$y, X, family = "lognormal", alpha = a, lambda = lam, tol = 1e-12)
  g <- glmnet::glmnet(X, ly, alpha = lam * a / (lam * (a + (1 - a) * sly)),
                      lambda = lam * (a + (1 - a) * sly))
  stopifnot(max(abs(fit$gamma[-1] - as.numeric(stats::coef(g))[-1])) < 1e-6)
}

## Gamma and Weibull fits: selection, methods and predictions
for (fam in c("gamma", "weibull")) {
  d <- generate_modal_data(X, c(1, 0.8, -0.6, rep(0, p - 2)), phi = 0.2, family = fam)
  fit <- modal_net(d$y, X, family = fam)
  stopifnot(inherits(fit, "ModalNet"), all(c(1, 2) %in% fit$selected),
            all(fit$path$converged), fit$phi > 0,
            length(predict(fit, X[1:5, ])) == 5,
            isTRUE(all.equal(predict(fit, type = "mean"), fit$mean_const * fitted(fit))),
            inherits(logLik(fit), "logLik"))
  s <- summary(fit)
  stopifnot(inherits(s, "summary.ModalNet"), nrow(s$coefficients) == length(fit$selected) + 1)
}

## Ridge keeps every covariate and converges
fit <- modal_net(dat$y, X, family = "gamma", alpha = 0)
stopifnot(length(fit$selected) == p, all(fit$path$converged))
