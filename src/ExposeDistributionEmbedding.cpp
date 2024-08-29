#include <RcppEigen.h>
#include <Rcpp.h>
#include "DistributionEmbedding"
#include "CholeskyDecomposition"
#include "KernelMatrix"
#include "KernelBasis"

RCPP_MODULE(RcppDistributionEmbedding) {
    using namespace RRCA::DISTRIBUTIONEMBEDDING;
    
    using MatrixType = RRCA::Matrix;
    using KernelMatrixType = RRCA::KernelMatrix<RRCA::GaussianKernel, MatrixType>;
    using CholeskyType = RRCA::PivotedCholeskyDecomposition<KernelMatrixType>;
    using KernelBasisType = RRCA::KernelBasis<KernelMatrixType>;
    using DistributionEmbeddingType = DistributionEmbedding<KernelMatrixType, CholeskyType, KernelBasisType>;

    Rcpp::class_<DistributionEmbeddingType>("DistributionEmbedding")
        .constructor<MatrixType, MatrixType>() 
        .method("condExpfVec", &DistributionEmbeddingType::condExpfVec)
        .method("solveUnconstrained", &DistributionEmbeddingType::solveUnconstrained)
    ;
}

