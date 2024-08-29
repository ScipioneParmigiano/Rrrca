library(Rrrca)
library(Matrix)

l1 <- 0.8
l2 <- 0.7
tol <- 0.001
lam <- 0.01

mean <- c(0, 0, 0)

set.seed(1) 
# correldis <- runif(1, min = -1.0, max = 1.0)
# rhoXY <- correldis
# rhoXZ <- correldis
# rhoYZ <- correldis

# sigX <- 1
# sigY <- 1
# sigZ <- 1

# covar <- matrix(c(
#   sigX * sigX, sigX * sigY * rhoXY, sigX * sigZ * rhoXZ,
#   sigX * sigY * rhoXY, sigY * sigY, sigY * sigZ * rhoYZ,
#   sigX * sigZ * rhoXZ, sigY * sigZ * rhoYZ, sigZ * sigZ
# ), nrow = 3, byrow = TRUE)

covar <- matrix(c(
  1, -0.736924, -0.0826997,
  -0.736924, 1, -0.562082,
  -0.0826997, -0.562082, 1
), nrow = 3, byrow = TRUE)

eigen_result <- eigen(covar)
eigvals <- eigen_result$values
eigvecs <- eigen_result$vectors

chol <- eigvecs %*% diag(sqrt(eigvals)) %*% t(eigvecs)

uniMatr <- matrix(c(
  0.459645, 0.0221783, 0.191117, 0.767538, 1.15418, 0.0820361, -1.77261, -0.718471, 0.749513, -0.0504267,
  0.189132, -0.233666, -0.443232, 1.30985, -0.773168, -0.162637, -0.175362, -0.0670528, -0.209844, -0.456606,
  -0.531634, 0.0140969, 0.0480184, 1.00209, -0.138363, 0.983943, 0.00739487, -2.01456, -0.638118, 0.329326
), nrow = 3, byrow = TRUE)

train <- chol %*% uniMatr 

print(train)
xtrain = as.matrix(train[2:3, ])
ytrain = as.matrix(train[1, ])

embedding <- new(DistributionEmbedding, xtrain, ytrain)

result <- embedding$solveUnconstrained(l1, l2, tol, lam)

print("res")
print(result)
