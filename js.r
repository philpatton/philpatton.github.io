# Define parameter values
n.occasions <- 7 # Number of capture occasions
N <- 400 # Superpopulation size
phi <- rep(0.7, n.occasions - 1) # Survival probabilities
b <- c(0.34, rep(0.11, n.occasions - 1)) # Entry probabilities
p <- rep(0.5, n.occasions) # Capture probabilities

PHI <- matrix(
    rep(phi, (n.occasions - 1) * N),
    ncol = n.occasions - 1,
    nrow = N,
    byrow = T
)
P <- matrix(rep(p, n.occasions * N), ncol = n.occasions, nrow = N, byrow = T)
# Function to simulate capture-recapture data under the JS model
simul.js <- function(PHI, P, b, N) {
    B <- rmultinom(1, N, b) # Generate no. of entering ind. per occasion
    n.occasions <- dim(PHI)[2] + 1
    CH.sur <- CH.p <- matrix(0, ncol = n.occasions, nrow = N)
    # Define a vector with the occasion of entering the population
    ent.occ <- numeric()
    for (t in 1:n.occasions) {
        ent.occ <- c(ent.occ, rep(t, B[t]))
    }
    # Simulate survival
    for (i in 1:N) {
        CH.sur[i, ent.occ[i]] <- 1 # Write 1 when ind. enters the pop.
        if (ent.occ[i] == n.occasions) {
            next
        }
        for (t in (ent.occ[i] + 1):n.occasions) {
            # Bernoulli trial: has individual survived occasion?
            sur <- rbinom(1, 1, PHI[i, t - 1])
            ifelse(sur == 1, CH.sur[i, t] <- 1, break)
        } #t
    } #i
    # Simulate capture
    for (i in 1:N) {
        CH.p[i, ] <- rbinom(n.occasions, 1, P[i, ])
    } #i
    # Full capture-recapture matrix
    CH <- CH.sur * CH.p
    # Remove individuals never captured
    cap.sum <- rowSums(CH)
    never <- which(cap.sum == 0)
    CH <- CH[-never, ]
    Nt <- colSums(CH.sur)
    return(list(CH = CH, B = B, N = Nt))
}
# Actual population size
# Execute simulation function
sim <- simul.js(PHI, P, b, N)
CH <- sim$CH

write.table(CH, 'sim-ch.txt', row.names = FALSE, col.names = FALSE)

# Add dummy occasion
CH.du <- cbind(rep(0, dim(CH)[1]), CH)
# Augment data
nz <- 500
CH.ms <- rbind(CH.du, matrix(0, ncol = dim(CH.du)[2], nrow = nz))
# Recode CH matrix: a 0 is not allowed in WinBUGS!
CH.ms[CH.ms == 0] <- 2 # Not seen = 2, seen = 1

# Bundle data
bugs.data <- list(y = CH.ms, n.occasions = dim(CH.ms)[2], M = dim(CH.ms)[1])
# Initial values
inits <- function() {
    list(
        mean.phi = runif(1, 0, 1),
        mean.p = runif(1, 0, 1),
        z = cbind(rep(NA, dim(CH.ms)[1]), CH.ms[, -1])
    )
}
# Parameters monitored
parameters <- c("mean.p", "mean.phi", "b", "Nsuper", "N", "B")
# MCMC settings
ni <- 20000
nt <- 3
nb <- 5000
nc <- 3
# Call WinBUGS from R (BRT 32 min)
js.ms <- bugs(
    bugs.data,
    inits,
    parameters,
    "js.bug",
    n.chains = nc,
    n.thin = nt,
    n.iter = ni,
    n.burnin = nb,
    debug = TRUE,
    bugs.directory = bugs.dir,
    working.directory = getwd()
)
print(js.ms, digits = 3)
