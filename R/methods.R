#' @export
print.ModalNet <- function(x, digits = 4, ...) {
  cat("Penalized modal regression (proximal Newton + glmnet)\n")
  cat(sprintf("Family: %s   n = %d   p = %d   penalty: %s   EBIC gamma = %.2f\n",
              x$family, x$n, x$p,
              if (x$alpha == 1) "Lasso" else sprintf("elastic net (alpha = %.2f)", x$alpha),
              x$gamma_ebic))
  cat(sprintf("Selected lambda = %.5f   selected covariates: %d of %d\n",
              x$lambda, length(x$selected), x$p))
  nz <- c(TRUE, x$gamma[-1] != 0)
  cat("\nCoefficients (log-mode scale):\n")
  print(round(x$gamma[nz], digits))
  cat(sprintf("\nDispersion phi = %.4f   log-likelihood = %.3f   EBIC = %.3f\n",
              x$phi, x$loglik, x$ebic))
  if (isTRUE(x$boundary))
    cat("Note: phi is at the boundary of its range; the fitted density is close to the\n",
        "exponential limit (mode near zero). Consider another family.\n", sep = "")
  invisible(x)
}

#' @export
summary.ModalNet <- function(object, digits = 4, ...) {
  nz <- which(object$gamma[-1] != 0)
  tab <- cbind(Estimate = object$gamma[c(1, nz + 1)],
               `exp(Estimate)` = exp(object$gamma[c(1, nz + 1)]))
  rownames(tab) <- names(object$gamma)[c(1, nz + 1)]
  out <- list(family = object$family, n = object$n, p = object$p,
              coefficients = tab, phi = object$phi, lambda = object$lambda,
              loglik = object$loglik, ebic = object$ebic, df = object$df,
              selected = object$xnames[nz], mean_const = object$mean_const,
              mean_intercept = object$gamma[1] + log(object$mean_const),
              path_summary = data.frame(lambda = object$path$lambda,
                                        df = object$path$df - 2,
                                        loglik = object$path$loglik,
                                        EBIC = object$path$ebic,
                                        iter = object$path$iter,
                                        converged = object$path$converged))
  class(out) <- "summary.ModalNet"
  out
}

#' @export
print.summary.ModalNet <- function(x, digits = 4, ...) {
  cat("--- Penalized Modal Regression Summary ---\n")
  cat(sprintf("Family: %s   n = %d   p = %d\n", x$family, x$n, x$p))
  cat(sprintf("EBIC-selected lambda = %.5f   active set size = %d\n\n", x$lambda, length(x$selected)))
  cat("Coefficients at the selected lambda (log link on the conditional mode):\n")
  print(round(x$coefficients, digits))
  cat(sprintf("\nDispersion phi: %.4f\n", x$phi))
  cat(sprintf("Log-likelihood: %.3f   EBIC: %.3f   df: %d\n", x$loglik, x$ebic, x$df))
  cat(sprintf("\nMode-mean relation: E[Y|x] = %.4f * M(x). The conditional mean has the same slopes\n", x$mean_const))
  cat(sprintf("and intercept %.4f (= %.4f + log %.4f) on the log scale.\n",
              x$mean_intercept, x$coefficients[1, 1], x$mean_const))
  invisible(x)
}

#' @export
coef.ModalNet <- function(object, s = NULL, ...) {
  if (is.null(s)) return(object$gamma)
  l <- which.min(abs(object$path$lambda - s))
  object$path$coef[, l]
}

#' Predict conditional modes (or means) from a fitted ModalNet model
#'
#' @param object a \code{"ModalNet"} object.
#' @param newdata covariate matrix with the same columns as the training
#'   \code{X}; if missing, the training matrix is used.
#' @param type \code{"mode"} (default) returns \eqn{\hat M}, \code{"link"} the
#'   linear predictor \eqn{\log \hat M}, and \code{"mean"} the implied
#'   conditional mean \eqn{c(\hat\phi)\hat M}.
#' @param ... unused.
#' @export
predict.ModalNet <- function(object, newdata, type = c("mode", "link", "mean"), ...) {
  type <- match.arg(type)
  X <- if (missing(newdata)) object$X else as.matrix(newdata)
  eta <- as.numeric(cbind(1, X) %*% object$gamma)
  switch(type, link = eta, mode = exp(eta), mean = object$mean_const * exp(eta))
}

#' @export
fitted.ModalNet <- function(object, ...) object$fitted.mode

#' @export
logLik.ModalNet <- function(object, ...) {
  structure(object$loglik, df = object$df, nobs = object$n, class = "logLik")
}

#' Plot the EBIC curve and the coefficient path of a ModalNet fit
#'
#' @param x a \code{"ModalNet"} object.
#' @param which \code{1} for the EBIC curve, \code{2} for the coefficient path,
#'   \code{c(1, 2)} (default) for both side by side.
#' @param ... further graphical arguments (ignored).
#' @export
plot.ModalNet <- function(x, which = c(1, 2), ...) {
  ll <- log(x$path$lambda)
  if (length(which) == 2) { op <- graphics::par(mfrow = c(1, 2)); on.exit(graphics::par(op)) }
  if (1 %in% which) {
    graphics::plot(ll, x$path$ebic, type = "b", pch = 16, xlab = expression(log(lambda)),
                   ylab = "EBIC", main = sprintf("EBIC (gamma = %.2f)", x$gamma_ebic))
    graphics::abline(v = log(x$lambda), lty = 2, col = "grey40")
  }
  if (2 %in% which) {
    cf <- x$path$coef[-1, , drop = FALSE]
    graphics::matplot(ll, t(cf), type = "l", lty = 1, xlab = expression(log(lambda)),
                      ylab = "coefficient", main = "Coefficient path (log-mode scale)")
    graphics::abline(h = 0, col = "grey70"); graphics::abline(v = log(x$lambda), lty = 2, col = "grey40")
    nz <- which(x$gamma[-1] != 0)
    if (length(nz)) graphics::text(min(ll), cf[nz, length(ll)], x$xnames[nz], pos = 4, cex = 0.7)
  }
  invisible(x)
}
