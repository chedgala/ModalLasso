## Family-specific pieces of the modal regression model (log link, M_i = exp(eta_i)).
## Each family provides:
##   loglik(y, eta, phi)      full log-likelihood (sum over i)
##   zw(y, eta, phi)          pseudo-response z_i and working weight w_i of the
##                            quadratic surrogate (Galarza & Lachos, Section 3.5)
##   update_phi(y, eta, phi)  maximizer of the profile log-likelihood in phi
##   mean_const(phi)          c(phi) such that E[Y | x] = c(phi) * M
##   check_phi(phi)           admissible range

.clamp_eta <- function(eta, bound = 30) pmin(pmax(eta, -bound), bound)
.phi_bounds <- c(1e-4, 1e4)
.clamp_phi <- function(phi) min(max(phi, .phi_bounds[1]), .phi_bounds[2])

## ---- Gamma: shape alpha = 1/phi + 1, scale beta_i = phi * M_i ---------------
.gamma_loglik <- function(y, eta, phi) {
  M <- exp(eta); alpha <- 1 / phi + 1
  sum((alpha - 1) * log(y) - y / (phi * M) - alpha * log(phi * M) - lgamma(alpha))
}
.gamma_zw <- function(y, eta, phi) {
  M <- exp(eta)
  w <- y / (phi * M)
  z <- eta + 1 - (1 + phi) * M / y
  list(z = z, w = w)
}
.gamma_update_phi <- function(y, eta, phi) {
  f <- function(lp) -.gamma_loglik(y, eta, exp(lp))
  .clamp_phi(exp(stats::optimize(f, interval = log(.phi_bounds))$minimum))
}
.gamma_mean_const <- function(phi) 1 + phi

## ---- Weibull: shape k = phi + 1.01, scale lambda_i = M_i (k/(k-1))^(1/k) -----
.weibull_loglik <- function(y, eta, phi) {
  k <- phi + 1.01
  lam <- exp(eta) * (k / (k - 1))^(1 / k)
  sum(log(k) - k * log(lam) + (k - 1) * log(y) - (y / lam)^k)
}
.weibull_zw <- function(y, eta, phi) {
  k <- phi + 1.01
  lam <- exp(eta) * (k / (k - 1))^(1 / k)
  r <- (y / lam)^k
  w <- k^2 * r
  z <- eta + (1 - 1 / r) / k
  list(z = z, w = w)
}
.weibull_update_phi <- function(y, eta, phi) {
  f <- function(lp) -.weibull_loglik(y, eta, exp(lp))
  .clamp_phi(exp(stats::optimize(f, interval = log(.phi_bounds))$minimum))
}
.weibull_mean_const <- function(phi) {
  k <- phi + 1.01
  (k / (k - 1))^(1 / k) * gamma(1 + 1 / k)
}

## ---- Lognormal: mu_i = eta_i + phi, sigma^2 = phi -----------------------------
.lognormal_loglik <- function(y, eta, phi) {
  ly <- log(y)
  sum(-0.5 * log(2 * pi * phi) - ly - (ly - eta - phi)^2 / (2 * phi))
}
.lognormal_zw <- function(y, eta, phi) {
  list(z = log(y) - phi, w = rep(1 / phi, length(y)))
}
.lognormal_update_phi <- function(y, eta, phi) {
  r2 <- mean((log(y) - eta)^2)
  (-1 + sqrt(1 + 4 * r2)) / 2
}
.lognormal_mean_const <- function(phi) exp(1.5 * phi)

## Joint update of the (unpenalized) intercept and the dispersion given the
## slopes. In the mode parameterization the intercept and phi are strongly
## coupled (for the Weibull, lambda_i = M_i {k/(k-1)}^{1/k}; for the Gamma,
## beta_i = phi M_i), so updating phi alone can stall on a ridge where
## the fit degenerates towards the exponential limit (phi -> 0 or phi -> Inf
## with M -> 0). Maximizing the log-likelihood over (gamma_0, phi) jointly,
## with the slopes fixed, is still block coordinate ascent on the penalized
## objective (the intercept carries no penalty) and removes the trap.
.update_intercept_phi <- function(fam, y, eta_slopes, gamma0, phi) {
  if (fam$name == "lognormal") {
    r <- log(y) - eta_slopes
    phi_new <- mean((r - mean(r))^2)
    return(list(gamma0 = mean(r) - phi_new, phi = phi_new))
  }
  lb <- log(.phi_bounds)
  f <- function(par) {
    if (par[2] < lb[1] || par[2] > lb[2] || abs(par[1]) > 30) return(1e100)
    -fam$loglik(y, .clamp_eta(eta_slopes + par[1]), exp(par[2]))
  }
  start <- c(gamma0, min(max(log(phi), lb[1] + 1e-6), lb[2] - 1e-6))
  o <- stats::optim(start, f, method = "Nelder-Mead",
                    control = list(maxit = 400, reltol = 1e-10))
  if (!is.finite(o$value) || o$value > f(start)) return(list(gamma0 = gamma0, phi = .clamp_phi(phi)))
  list(gamma0 = o$par[1], phi = .clamp_phi(exp(o$par[2])))
}

.get_family <- function(family) {
  switch(family,
    gamma = list(name = "gamma", loglik = .gamma_loglik, zw = .gamma_zw,
                 update_phi = .gamma_update_phi, mean_const = .gamma_mean_const,
                 phi_label = "phi (alpha = 1/phi + 1)"),
    weibull = list(name = "weibull", loglik = .weibull_loglik, zw = .weibull_zw,
                   update_phi = .weibull_update_phi, mean_const = .weibull_mean_const,
                   phi_label = "phi (k = phi + 1.01)"),
    lognormal = list(name = "lognormal", loglik = .lognormal_loglik, zw = .lognormal_zw,
                     update_phi = .lognormal_update_phi, mean_const = .lognormal_mean_const,
                     phi_label = "phi (sigma^2)"),
    stop("family must be one of 'gamma', 'weibull', 'lognormal'"))
}
