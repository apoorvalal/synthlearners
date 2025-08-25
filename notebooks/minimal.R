library(CVXR); library(data.table)

# %% functions
# reshape panel data from long to wide for factor models / outcomes
panelMatrices = function(dt, unit_id, time_id, treat, outcome) {
  dt = as.data.table(dt)
  # function to extract first column, convert it to rownames for a matrix
  matfy = function(X) {
    idnames = as.character(X[[1]])
    X2 = as.matrix(X[, -1])
    rownames(X2) = idnames
    X2
  }
  # reshape formula
  fmla = as.formula(paste0(unit_id, "~", time_id))
  # treatment matrix
  kv = c(unit_id, time_id, treat)
  W = matfy(dcast(dt[, ..kv], fmla, value.var = treat))
  # outcome matrix
  kv = c(unit_id, time_id, outcome)
  Y = matfy(dcast(dt[, ..kv], fmla, value.var = outcome))
  # move treated units to bottom of W and Y matrix
  treatIDs = which(rowSums(W) > 1)
  W = rbind(W[-treatIDs, ], W[treatIDs, , drop = FALSE])
  Y = rbind(Y[-treatIDs, ], Y[treatIDs, , drop = FALSE])
  N0 = nrow(W) - length(treatIDs)
  T0 = min(which(colSums(W) > 0)) - 1
  list(W = W, Y = Y, N0 = N0, T0 = T0)
}

# %% optimisation functions
ε = 1e-6
σ = \(Y, N0, T0) sd(apply(Y[1:N0, 1:T0], 1, diff))

# return w minimizing ||Aw - b||^2      + ζ^2 n || w ||^2     if intercept=FALSE
#      | w minimizing ||Aw + w0 - b||^2 + ζ^2 n || w ||^2     if intercept=TRUE
# here n = length(b)

simplexLeastSquares = function(A, b, ζ = 0, intercept = FALSE, solv = 'MOSEK') {
  w = CVXR::Variable(ncol(A))
  constraints = list(sum(w) == 1, w >= 0)
  if (intercept) {
    w0 = CVXR::Variable(1)
    objective = sum((A %*% w + w0 - b)^2) + ζ^2 * length(b) * sum(w^2)
  } else {
    objective = sum((A %*% w - b)^2) + ζ^2 * length(b) * sum(w^2)
  }
  cvx.problem = CVXR::Problem(CVXR::Minimize(objective), constraints)
  cvx.output = CVXR::solve(cvx.problem, solver = solv)
  as.numeric(cvx.output$getValue(w))
}

sC = function(Y, N0, T0, ζ.ω = 1e-6 * σ(Y, N0, T0)) {
  N = nrow(Y); T = ncol(Y); N1 = N - N0; T1 = T - T0;
  ω = simplexLeastSquares(t(Y[1:N0, 1:T0]),
    colMeans(Y[(N0 + 1):N, 1:T0, drop = FALSE]),
    ζ = ζ.ω,
    intercept = FALSE
  )
  estimate = t(c(-ω, rep(1 / N1, N1))) %*% Y %*% c(-rep(0, T0), rep(1 / T1, T1))
}

dID = function(Y, N0, T0) {
  N = nrow(Y); T = ncol(Y); N1 = N - N0; T1 = T - T0;
  estimate = (t(c(-rep(1 / N0, N0), rep(1 / N1, N1))) %*%
    Y %*%
    c(-rep(1 / T0, T0), rep(1 / T1, T1))
  )
}

sDiD = function(Y, N0, T0,
                ζ.ω = ((nrow(Y) - N0) * (ncol(Y) - T0))^(1 / 4) * σ(Y, N0, T0)) {
  N = nrow(Y); T = ncol(Y); N1 = N - N0; T1 = T - T0;
  λ = simplexLeastSquares(Y[1:N0, 1:T0],
    rowMeans(Y[1:N0, (T0 + 1):T, drop = FALSE]),
    ζ = ε * σ(Y, N0, T0),
    intercept = TRUE
  )
  ω = simplexLeastSquares(t(Y[1:N0, 1:T0]),
    colMeans(Y[(N0 + 1):N, 1:T0, drop = FALSE]),
    ζ = ζ.ω,
    intercept = TRUE
  )
  estimate = t(c(-ω, rep(1 / N1, N1))) %*% Y %*% c(-λ, rep(1 / T1, T1))
}
# %% data
prop99 = fread("california_prop99.csv")
setup = panelMatrices(prop99,
  unit_id = "State", time_id = "Year",
  treat = "treated", outcome = "PacksPerCapita"
)

# %%
sDiD(setup$Y, setup$N0, setup$T0) %>% print
sC(setup$Y, setup$N0, setup$T0) %>% print
dID(setup$Y, setup$N0, setup$T0) %>% print
# %%
(sdid_ests = list(
  synthdid::synthdid_estimate(setup$Y, setup$N0, setup$T0),
  synthdid::sc_estimate(setup$Y, setup$N0, setup$T0),
  synthdid::did_estimate(setup$Y, setup$N0, setup$T0)
))

# %%
lapply(sdid_ests, plot)
# %%
