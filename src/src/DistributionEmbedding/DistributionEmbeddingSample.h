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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGSAMPLE_H_
#define RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGSAMPLE_H_



namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{
    
//     kernel JDL asymptotics version
//     learns the RN derivative dQ/dP
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class DistributionEmbeddingSample
{
// zeta is the sample from P and z is the sample from Q
public:
    DistributionEmbeddingSample ( const Matrix& qdata_,const Matrix& pdata_) :
        nzeta(pdata_.cols()),
        nz(qdata_.cols()),
        zeta ( pdata_ ),
        z ( qdata_ ),
        Kzeta ( zeta ),
        baszeta(Kzeta,pivzeta),
        Kz ( z ),
        basz(Kz,pivz){
    }
    


    unsigned int getSubspaceDimension() const {
        return(Lzeta.cols());
    }
    

    
    
    
          /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
        int solveUnconstrained ( double l, double prec,double lam )
    {
        
        precomputeKernelMatrices ( l, prec,lam );
        const double nn(Lzeta.rows());
        L_tL = Lzeta.transpose()*Lzeta/nn;
        toInvert = L_tL;
        toInvert.diagonal().array()+=lam;
        u_til_lambda = Uzeta * toInvert.llt().solve(Lzeta.transpose() *(-q_til_lambda-Vector::Constant(Lzeta.rows(),p)) )/nn;

        return ( EXIT_SUCCESS );
    }
    
              /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
        int solve ( double l, double prec,double lam )
    {
        
        return(solveUnconstrained(l,prec,lam));
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
//     Matrix condExpfVec ( const Matrix& Xs, bool structuralYes = false ) const    {
// //                 compute a block matrix where the Xs are on the column dimension and the Ys are on the row dim 
//         Matrix hblock = Matrix::Constant(yint.cols(), Xs.cols(),p);
//         Matrix arg(yint.rows() + Xs.rows(), Xs.cols());
//         Matrix zetas;
//         Matrix zs;
//         
// //      
//         // for(auto j = 0; j < Xs.cols(); ++j){
//         //     for(auto i = 0; i < yint.cols(); ++i){
//         //         arg << Xs.col(j),yint.col(i);
//         //         zetas = baszeta.eval(arg);
//         //         zs = basz.eval(arg);
//         //         hblock(i,j) += zetas.dot(u_til_lambda) + zs.dot(lambda_vec);
//         //     }
//         // }
//         // construct kernel matrices first
//         // for(auto j = 0; j < Xs.cols(); ++j){
//             for(auto i = 0; i < yint.cols(); ++i){
//                 arg << Xs ,yint.col(i).replicate(1,Xs.cols());
//                 zetas = baszeta.eval(arg);
//                 zs = basz.eval(arg);
//                 hblock.row(i) += (zetas * u_til_lambda +  zs * lambda_vec).transpose();
//             }
//         // }
//         
//         // std::cout << "hblock " << std::endl  << hblock << std::endl;
//         if(structuralYes){
//             hblock.array().rowwise() /= hblock.cwiseMax(0.0).colwise().sum().array();
//             return(hblock.cwiseMax(0.0));
//         }
//         return(hblock/static_cast<double>((yint.cols())));
//         
//     }
    
                /*
    *    \brief computes conditional expectation of all the rows in Ys
    */
//     const Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs, bool structuralYes = false ) const {
//         Matrix hblock = Matrix::Constant(yint.cols(), Xs.cols(),p);
//         Matrix arg(yint.rows() + Xs.rows(), Xs.cols());
//         Matrix zetas;
//         Matrix zs;
//         // Vector arg(zeta.rows());
//         // RowVector zetas;
//         // RowVector zs;
//         // for(auto j = 0; j < Xs.cols(); ++j){
//         //     for(auto i = 0; i < yint.cols(); ++i){
//         //         arg << Xs.col(j),yint.col(i);
//         //         zetas = baszeta.eval(arg);
//         //         zs = basz.eval(arg);
//         //         hblock(i,j) += zetas.dot(u_til_lambda) + zs.dot(lambda_vec);
//         //     }
//         // }
//         for(auto i = 0; i < yint.cols(); ++i){
//                 arg << Xs ,yint.col(i).replicate(1,Xs.cols());
//                 zetas = baszeta.eval(arg);
//                 zs = basz.eval(arg);
//                 hblock.row(i) += (zetas * u_til_lambda +  zs * lambda_vec).transpose();
//             }
//         if(structuralYes){
//             hblock.array().rowwise() /= hblock.cwiseMax(0.0).colwise().sum().array();
//             return(Ys * hblock.cwiseMax(0.0));
//         }
//         
//         return(Ys * hblock/static_cast<double>((yint.cols())));
//     }
                        /*
    *    \brief returns the "realized objective function for the validatino sample"
    */
    double getScore(const Matrix& valq,const Matrix& valp,  bool structuralYes){
//         now we evaluate the objective function -2<J^*_{(X,Y)}1-J^*p_*,h>_H+<J^*Jh,h>_H at the sample points that are the function arguments
//      we do not need that valq and valp have the same length
        const Matrix z_zs = basz.eval(valq);
        const Matrix z_zetas = baszeta.eval(valq);
        
        const Matrix zeta_zs = basz.eval(valp);
        const Matrix zeta_zetas = baszeta.eval(valp);
        
        const Vector zvec    = (z_zetas * u_til_lambda +  z_zs * lambda_vec); // with respect to joint measure
        const Vector zetavec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec); // with respect to independence measure
        
        //      now that we have the sample, we can evaluate the objective function (that was solved before at these points)
        if(structuralYes){
            return(-2.0*zvec.cwiseMax(-p).mean()+2.0*p*zetavec.cwiseMax(-p).mean()+(zetavec.cwiseMax(-p).cwiseProduct(zetavec.cwiseMax(-p))).mean());
        }
        
        return(-2.0*zvec.mean()+2.0*p*zetavec.mean()+(zetavec.cwiseProduct(zetavec)).mean());
    }
 
//     compute this for the trainingssample 
    double getasymteststat(double lam,double l, double prec, bool structuralYes,  int order = -1){ 
        solveUnconstrained( l,  prec, lam);
        
        
        const unsigned int m = Lzeta.cols();
        const unsigned int lowRank = (order > 0 ? std::min(static_cast<unsigned int>(order),m-1) : m-1);

        
        const double ntd(Lzeta.rows());

        
        Vector hvec;
        if(structuralYes){
            hvec = (Kzeta_block * u_til_lambda +  Kzeta_z_block * lambda_vec).cwiseMax(-p);
        } else {
            hvec = (Kzeta_block * u_til_lambda +  Kzeta_z_block * lambda_vec);
        }
        Vector outerHelp = Lzeta.colwise().mean().transpose();
        Eigen::SelfAdjointEigenSolver<RRCA::Matrix> es (2.0*(L_tL - outerHelp*outerHelp.transpose()));
        Vector need = sqrt(ntd) * Lzeta.transpose() * hvec / ntd+sqrt(ntd) * lam * Uzeta.transpose() * hvec(pivzeta.pivots());
        double teststat(((es.eigenvectors().rightCols(lowRank).transpose() * need).array().square().matrix().cwiseQuotient(es.eigenvalues().tail(lowRank))).sum());
        nt = lowRank;
        
        return(teststat);
    }
    
      
    
    unsigned int getNt() const {
        return(nt);
    }

    

private:
    const unsigned int nzeta;
    const unsigned int nz;
    
    const Matrix& zeta;
    const Matrix& z;
    
    
    Matrix Lzeta; // the important ones
    Matrix Lz; // the important ones


    // Matrix Bzeta; // the basis transformation matrix
    Matrix Uzeta; // the basis transformation matrix
    // Matrix Bz; // the basis transformation matrix for the q lambda projector
    Matrix Uz; // the basis transformation matrix for the q lambda projector
    Vector q_til_lambda;
    Vector u_til_lambda;
    Matrix toInvert;
    Matrix L_tL;
    
    Matrix Kzeta_z_block;
    Matrix Kzeta_block;
    
    Vector lambda_vec;
    Matrix yint; // to compute the conditional expectations. iid sample
//     Matrix Q;
    
    // Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
    // Vector prob_vec; // the important ones
    unsigned int nt;

    KernelMatrix Kzeta;
    LowRank pivzeta;
    KernelBasis baszeta;
 
    KernelMatrix Kz;
    LowRank pivz;
    KernelBasis basz;
    

    
    double tol;
    static constexpr double p = 1.0;
    
    
    
//     these are the empirical bounds for the functions

//     /*
//    *    \brief computes the kernel basis and tensorizes it
//    */ 
    void precomputeKernelMatrices ( double l, double prec, double lambda){
        Kzeta.kernel().l = l;
        Kz.kernel().l = l;
        
        pivzeta.compute ( Kzeta, prec );
        baszeta.init(Kzeta, pivzeta.pivots());
        baszeta.initNewtonBasisWeights(pivzeta);
        Uzeta = baszeta.matrixU();
        Kzeta_block = baszeta.eval(zeta);
        Lzeta = Kzeta_block * Uzeta;
        
        
        pivz.compute ( Kz, prec );
        basz.init(Kz,pivz.pivots());
        basz.initNewtonBasisWeights(pivz);
        Uz = basz.matrixU();
        Lz = basz.eval(z) * Uz;
               
        lambda_vec = Uz  *  Lz.colwise().mean().transpose()/ lambda;
        
        Kzeta_z_block = basz.eval(zeta);
        
        q_til_lambda = Kzeta_z_block * lambda_vec;
    }
};


template<typename KernelMatrix, typename LowRank, typename KernelBasis>
constexpr double DistributionEmbeddingSample<KernelMatrix, LowRank, KernelBasis>::p;















} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
