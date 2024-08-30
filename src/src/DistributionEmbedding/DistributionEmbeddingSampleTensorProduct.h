////////////////////////////////////////////////////////////////////////////////
// This file is part of RRCA, the Roadrunner Covariance Analsys package.      //
//                                                                            //
// Copyright (c) 2021, Michael Multerer and Paul Schneider                    //
//                                                                            //
// All rights reserved.                                                       //
//                                                                            //
// This source code is subject to the BSD 3-clause license and without        //
// any warranty, see <https://github.com/muchip/RRCA> for further             //
// information.                                                               //
////////////////////////////////////////////////////////////////////////////////
#ifndef RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGSAMPLETENSORPRODUCT_H_
#define RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGSAMPLETENSORPRODUCT_H_



namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{
    
//     kernel JDL asymptotics version, here with a tensor product hypothesis space
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class DistributionEmbeddingSampleTensorProduct
{

public:
    DistributionEmbeddingSampleTensorProduct ( const Matrix& xdata_,const Matrix& ydata_) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        n((!static_cast<bool>(xdata_.cols() % 2) ? xdata_.cols()/2 : (xdata_.cols()-1)/2 )),
        zdata_x (xdata_.rows(),n),
        zdata_y (ydata_.rows(),n),
        zeta_x(xdata_.rows(),n),
        zeta_y(ydata_.rows(),n),
        Kzeta_x ( zeta_x ),
        Kzeta_y ( zeta_y ),
        baszeta_x(Kzeta_x,pivzeta_x),
        baszeta_y(Kzeta_y,pivzeta_y),
        Kz_x ( zdata_x ),
        Kz_y ( zdata_y ),
        basz_x(Kz_x,pivz_x),
        basz_y(Kz_y,pivz_y)
        {
            
            zdata_x         = xdata_.leftCols(n);
            zdata_y         = ydata_.leftCols(n);
            
            zeta_x          = xdata_(Eigen::all, Eigen::seq(0, (!static_cast<bool>(xdata_.cols() % 2) ? xdata_.cols()-1 : xdata_.cols()-2 ) , 2));
            zeta_y          = ydata_(Eigen::all, Eigen::seq(1, Eigen::last, 2) );
            
            iVector indices = iVector::LinSpaced(xdata_.cols(), 0, xdata_.cols() -1);
            // thread_local std::mt19937 eng(seed);
            // std::shuffle(indices.begin(), indices.end(),eng);
            // iid_int = indices.head(n);
            
            // std::cout << " Holla " << iid_int.head(3).transpose() << std::endl;
            
            // std::random_shuffle(indices.begin(), indices.end());
            
            iid_int = indices(Eigen::seq(1, Eigen::last, 2));
            yint = zeta_y;
            
    }

    // unsigned int getSubspaceDimension() const {
    //     return(Lzeta.cols());
    // }
    
        /*
   *    \brief returns the y values used for integration
   */
    const Matrix& get_y_int() const {
        return(yint);
    }
    
    
    
          /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
        int solveUnconstrained ( double lx, double ly, double prec,double lam )
    {
        
        precomputeKernelMatrices ( lx, ly, prec,lam );
        return ( EXIT_SUCCESS );
    }
    
        /*
   *    \brief returns the y values used for integration
   */

//     /*
//    *    \brief computes conditional expectation in one dimension in Y
//    */
//     Matrix condExpfY_X ( const std::vector<std::function<double ( double ) > >& funs, const Matrix& Xs ) const
//     {
//        
//                 
//          
//         return ( res );
//     }
//     
        /*
   *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
   */
    Matrix condExpfVec ( const Matrix& Xs, bool isPositive = true ) const    {
//                 compute a block matrix where the Xs are on the column dimension and the Ys are on the row dim 
        if(isPositive){
            const Matrix hblock = Matrix::Constant(yint.cols(), Xs.cols(),p) + K_zetazeta_y * U_lam * baszeta_x.eval(Xs).transpose() + K_zetaz_y * Q_lam * basz_x.eval(Xs).transpose();
        // std::cout << "hblock "<< hblock << std::endl;
        // hblock.array().rowwise() /= hblock.cwiseMax(0.0).colwise().sum().array();
            return((hblock.array().rowwise() / hblock.cwiseMax(0.0).colwise().sum().array()).cwiseMax(0.0));
        }
        return(Matrix::Constant(yint.cols(), Xs.cols(),p) + K_zetazeta_y * U_lam * baszeta_x.eval(Xs).transpose() + K_zetaz_y * Q_lam * basz_x.eval(Xs).transpose());
    }
    
                /*
    *    \brief computes conditional expectation of all the rows in Ys
    */
    const Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs, bool isPositive = true ) const {
        if(isPositive){
            const Matrix Ublock     = U_lam * baszeta_x.eval(Xs).transpose();
            const Matrix Qblock     = Q_lam * basz_x.eval(Xs).transpose();
            const RowVector norma   = (Matrix::Constant(yint.cols(), Xs.cols(),p) + K_zetazeta_y * Ublock + K_zetaz_y * Qblock).cwiseMax(0.0).colwise().sum();

            return(Ys * ((Matrix::Constant(yint.cols(), Xs.cols(),p)+K_zetazeta_y * Ublock + K_zetaz_y * Qblock).array().rowwise()/norma.array()).matrix().cwiseMax(0.0));
        }
        return(Ys * (Matrix::Constant(yint.cols(), Xs.cols(),p) + K_zetazeta_y * U_lam * baszeta_x.eval(Xs).transpose() + K_zetaz_y * Q_lam * basz_x.eval(Xs).transpose()));
    }

    

private:
    const Matrix &xdata;
    const Matrix &ydata;
    const unsigned int n;
    Matrix zdata_x;
    Matrix zdata_y;
    
    Matrix y_z;
    
    Matrix zeta_x;
    Matrix zeta_y;
    

    
    Matrix Lzeta_x; // the important ones
    Matrix Lz_x; // the important ones
    
    Matrix Lzeta_y; // the important ones
    Matrix Lz_y; // the important ones


    Matrix Qzeta_x; // the basis transformation matrix
    Matrix Bz_x; // the basis transformation matrix for the q lambda projector
    
    Matrix Qzeta_y; // the basis transformation matrix
    Matrix Bz_y; // the basis transformation matrix for the q lambda projector
    
    Vector q_til_lambda;
    Vector u_til_lambda;
    Matrix toInvert;
    Matrix L_tL;
    
    Matrix Q_lam;
    Matrix U_lam;
    
    Matrix K_zetaz_y;
    Matrix K_zetaz_x;
    Matrix K_zetazeta_y;
    // iVector iid_int; // the indices for the iid integration
    // Matrix yint; // to compute the conditional expectations. iid sample
//     Matrix Q;
    
    // Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
    // Vector prob_vec; // the important ones

    KernelMatrix Kzeta_x;
    KernelMatrix Kzeta_y;
    LowRank pivzeta_x;
    LowRank pivzeta_y;
    KernelBasis baszeta_x;
    KernelBasis baszeta_y;
 
    KernelMatrix Kz_x;
    KernelMatrix Kz_y;
    LowRank pivz_x;
    LowRank pivz_y;
    KernelBasis basz_x;
    KernelBasis basz_y;
    
    iVector iid_int; // the indices for the iid integration
    Matrix yint; // to compute the conditional expectations. iid sample
    

    
    double tol;
    static constexpr double p = 1.0;
    
    
    
//     these are the empirical bounds for the functions

//     /*
//    *    \brief computes the kernel basis and tensorizes it
//    */ 
    void precomputeKernelMatrices ( double lx, double ly, double prec, double lambda){
        Kzeta_x.kernel().l = lx;
        Kz_x.kernel().l = lx;
        double const nn(n);
        
        pivzeta_x.compute ( Kzeta_x, prec );
        baszeta_x.init(Kzeta_x, pivzeta_x.pivots());
        baszeta_x.initSpectralBasisWeights(pivzeta_x);
        Qzeta_x = baszeta_x.matrixQ() ;
        Lzeta_x = baszeta_x.eval(zeta_x) * Qzeta_x;
        
        Kzeta_y.kernel().l = ly;
        Kz_y.kernel().l = ly;
        
        pivzeta_y.compute ( Kzeta_y, prec );
        baszeta_y.init(Kzeta_y, pivzeta_y.pivots());
        baszeta_y.initSpectralBasisWeights(pivzeta_y);
        Qzeta_y = baszeta_y.matrixQ();
        K_zetazeta_y = baszeta_y.eval(zeta_y);
        Lzeta_y = K_zetazeta_y * Qzeta_y;
        

        
        pivz_x.compute ( Kz_x, prec );
        basz_x.init(Kz_x,pivz_x.pivots());

        basz_x.initNewtonBasisWeights(pivz_x);
        Bz_x = basz_x.matrixU();
        Lz_x = basz_x.eval(zdata_x) * Bz_x;
        
        pivz_y.compute ( Kz_y, prec );
        basz_y.init(Kz_y,pivz_y.pivots());
        basz_y.initNewtonBasisWeights(pivz_y);
        Bz_y = basz_y.matrixU();

        Lz_y = basz_y.eval(zdata_y) * Bz_y;
        
        K_zetaz_y = basz_y.eval(zeta_y);
        K_zetaz_x = basz_x.eval(zeta_x);
        

        
        Q_lam = nn/lambda *  Bz_y * Lz_y.colwise().mean().transpose() * Lz_x.colwise().mean() * Bz_x.transpose();
        const Matrix inter =  Lzeta_y.transpose() * (-(K_zetaz_y * Q_lam *K_zetaz_x.transpose()).array() - p).matrix() * Lzeta_x;
        U_lam = Qzeta_y * (inter.array()/((baszeta_y.vectorLambda() * baszeta_x.vectorLambda().transpose()/nn).array()+lambda)).matrix() * Qzeta_x.transpose()/nn;
        

        
    }
};


template<typename KernelMatrix, typename LowRank, typename KernelBasis>
constexpr double DistributionEmbeddingSampleTensorProduct<KernelMatrix, LowRank, KernelBasis>::p;















} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
