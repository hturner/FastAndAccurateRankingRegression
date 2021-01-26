library(reticulate)
library(Matrix)
np <- import("numpy")

dir <- "sushi_dectet_"

# Load data from .py file for now
rankings_train <- np$load(file.path("..",  "data", paste0(dir, "data"), "rankings.npy")) + 1
X <- np$load(file.path("..",  "data", paste0(dir, "data"), "features.npy"))
## sparse matrix object
mat_Pij <- np$load(file.path("..",  "data", paste0(dir, "data"), "mat_Pij.npz"))
mat_Pij <- sparseMatrix(i = c(mat_Pij$f["indices"]),
                        p = c(mat_Pij$f["indptr"]),
                        x = c(mat_Pij$f["data"]),
                        index1 = FALSE)
endog <- rankings_train[, 1]
exog <- rankings_train

# Initialization
inits <-  init_params(X, rankings_train, mat_Pij, method_beta_b_init = "QP")

log_dict <- list()
log_dict[["log_admm"]] <- ADMM_log$new(rankings_train, X,
                                       method_pi_tilde_init = "prev")
log_dict[["log_admm_conv"]] <- FALSE
log_dict[["beta_log_admm"]] <- inits$exp_beta_init
log_dict[["pi_log_admm"]] <- softmax(X %*% log_dict[["beta_log_admm"]])
log_dict[["u_log_admm"]] <- inits$u_init
log_dict[["time_log_admm"]] = inits$time_exp_beta_init + inits$time_u_init
log_dict[["diff_pi_log_admm"]] <- sqrt(sum(log_dict[["pi_log_admm"]]^2))
log_dict[["diff_beta_log_admm"]] <- sqrt(sum(log_dict[["beta_log_admm"]]^2))
log_dict[["prim_feas_log_admm"]] <- sqrt(sum((X %*% log_dict[["beta_log_admm"]] - log(log_dict[["pi_log_admm"]] + epsilon))^2))
log_dict[["dual_feas_log_admm"]] <- sqrt(sum((t(log_dict[["log_admm"]]$X) %*% log(log_dict[["pi_log_admm"]] + epsilon))^2))
log_dict[["obj_log_admm"]] <- objective(log_dict[["pi_log_admm"]], rankings_train)
log_dict['iter_log_admm'] <- 0

rho <- 1

n <- nrow(X)
for (iter in seq_len(n_iter)){
    # log_admm update
    if (!log_dict[["log_admm_conv"]]){
        log_dict[["pi_log_admm_prev"]] <- log_dict[["pi_log_admm"]]
        log_dict[["beta_log_admm_prev"]] <- log_dict[["beta_log_admm"]]
        log_dict[["tilde_pi_log_admm_prev"]] <- softmax(X %*% log_dict[["beta_log_admm"]])

        res <- log_dict[["log_admm"]]$fit_log(rho, weights = log_dict[["pi_log_admm"]],
                                              beta = log_dict[["beta_log_admm"]],
                                              u = log_dict[["u_log_admm"]])
        log_dict[["pi_log_admm"]] <- res$weights
        log_dict[["beta_log_admm"]] <- res$beta
        log_dict[["u_log_admm"]] <- res$u
        time_log_admm_iter <- res$time
        # scores predicted by beta
        log_dict[["tilde_pi_log_admm"]] <- softmax(X %*% log_dict[["beta_log_admm"]])
        log_dict[["time_log_admm"]] <- c(log_dict[["time_log_admm"]], time_log_admm_iter)
        log_dict[["diff_pi_log_admm"]] <- c(log_dict[["diff_pi_log_admm"]], sqrt(sum((log_dict[["pi_log_admm_prev"]] - log_dict[["pi_log_admm"]])^2)))
        log_dict[["diff_beta_log_admm"]] <- c(log_dict[["diff_beta_log_admm"]], sqrt(sum((log_dict[["beta_log_admm_prev"]] - log_dict[["beta_log_admm"]])^2)))
        log_dict[["prim_feas_log_admm"]] <- c(log_dict[["prim_feas_log_admm"]], sqrt(sum((X %*% log_dict[["beta_log_admm"]] - log(log_dict[["pi_log_admm"]] + epsilon))^2)))
        log_dict[["dual_feas_log_admm"]] <- c(log_dict[["dual_feas_log_admm"]], sqrt(sum((t(log_dict[["log_admm"]]$X) %*% (log(log_dict[["pi_log_admm_prev"]] + epsilon) - log(log_dict[["pi_log_admm"]] + epsilon)))^2)))
        log_dict[["obj_log_admm"]] <- c(log_dict[["obj_log_admm"]], objective(log_dict[["pi_log_admm"]], rankings_train))
        log_dict[["iter_log_admm"]] <- log_dict[["iter_log_admm"]] + 1
        log_dict[["log_admm_conv"]] <- sqrt(sum((log_dict[["pi_log_admm_prev"]] - log_dict[["pi_log_admm"]])^2)) < rtol * sqrt(sum(log_dict[["pi_log_admm"]]^2)) &&
            sqrt(sum((log_dict[["tilde_pi_log_admm_prev"]] - log_dict[["tilde_pi_log_admm"]])^2)) < rtol * sqrt(sum(log_dict[["tilde_pi_log_admm"]]^2))
    }
    # stop if converged
    if (log_dict[["log_admm_conv"]]) {
        break
    }
}

# Correct time scale
log_dict[["time_cont_log_admm"]] <- vapply(seq_along(log_dict[["time_log_admm"]]), function(ind) sum(log_dict[["time_log_admm"]][1:ind]), numeric(1))

# Save results as R object
saveRDS(log_dict, file.path("..",  "results", paste0(dir, "data"), "_logs_pladmm.rds"))

object <- readRDS(file.path("..",  "results", paste0(dir, "data"), "_logs_pladmm.rds"))
object["log_admm_conv"]
round(object[["beta_log_admm"]], 3) # coefficients
round(object[["pi_log_admm"]], 3) # implied worth = worth from standard PL model
