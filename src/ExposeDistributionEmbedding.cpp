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
        // .constructor<const Eigen::MatrixXd&, const Eigen::MatrixXd&>() // Adjust based on constructor
        // .constructor([](const Rcpp::NumericMatrix& rX, const Rcpp::NumericMatrix& rY) {
        //     Eigen::MatrixXd X = Rcpp::as<Eigen::MatrixXd>(rX);
        //     Eigen::MatrixXd Y = Rcpp::as<Eigen::MatrixXd>(rY);
        //     return new DistributionEmbeddingType(X, Y);
        // })

    // Rcpp::class_<DistributionEmbedding<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>, 
    //             RRCA::PivotedCholeskyDecomposition<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>,
    //             RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>>>("DistributionEmbedding")
    //     .constructor([](const Rcpp::NumericMatrix& rX, const Rcpp::NumericMatrix& rY) {
    //             Eigen::MatrixXd X = Rcpp::as<Eigen::MatrixXd>(rX);
    //             Eigen::MatrixXd Y = Rcpp::as<Eigen::MatrixXd>(rY);
    //             return new DistributionEmbedding<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>, 
    //             RRCA::PivotedCholeskyDecomposition<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>,
    //             RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>>(X, Y);
    //         })
    //     // .method("condExpfVec", &DistributionEmbedding<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>, RRCA::PivotedCholeskyDecomposition<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>, RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>>::condExpfVec)
    //     // .method("solveUnconstrained", &DistributionEmbedding<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>, RRCA::PivotedCholeskyDecomposition<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>, RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>>::solveUnconstrained)
    ;
}

