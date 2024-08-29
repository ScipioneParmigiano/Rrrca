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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDING_H_
#define RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDING_H_

#include "../util/Stopwatch.h"

namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{
    
//     plain vanilla implementation, to be used only with SumKernel
//     !!! to be used only with SumKernel !!!
template<typename KernelMatrix, typename LowRank, typename KernelBasis,typename KernelMatrixY = KernelMatrix, typename LowRankY = LowRank, typename KernelBasisY = KernelBasis>
class DistributionEmbedding
{
    typedef typename KernelMatrix::value_type value_type;


private:    
    /*
   *    \brief computes the kernel basis and tensorizes it
   */ 
    void precomputeKernelMatrices ( double l1, double l2,double prec, double lam )
    {
//         RRCA::IO::Stopwatch sw;
//         sw.tic();
        std::cout << 11 << std::endl;
        std::cout << "kx precompute 1 " << Kx.full() << std::endl; // per qualche ragione kx scompare
        Kx.kernel().l = l1;
        std::cout << 11.5 << std::endl;
        Ky.kernel().l = l2;
        std::cout << 11.55 << std::endl;

        std::cout << "kx precompute" << Kx.full() << std::endl; // fallisce qui
        std::cout << 12 << std::endl;
        pivx.compute ( Kx,prec);
        std::cout << 13 << std::endl;
        basx.init(Kx, pivx.pivots());
        std::cout << 14 << std::endl;
        basx.initSpectralBasisWeights(pivx);
        std::cout << 15 << std::endl;
        
        
        Qx =  basx.matrixQ() ;
        std::cout << 16 << std::endl;
        Kxblock = basx.eval(xdata);
        
        std::cout << 16.5 << std::endl;

        LX = Kxblock * Qx;
        std::cout << 17 << std::endl;
        
        pivy.compute ( Ky,prec);
        basy.init(Ky, pivy.pivots());
        basy.initSpectralBasisWeights(pivy);
        

        
        Qy =  basy.matrixQ() ;
        Kyblock = basy.eval(ydata);
        

        LY = Kyblock * Qy;
        
        xbound_min = LX.colwise().minCoeff().transpose();
        xbound_max = LX.colwise().maxCoeff().transpose();
        
        
        ybound_min = LY.colwise().minCoeff().transpose();
        ybound_max = LY.colwise().maxCoeff().transpose();
        
        
//      bounds
        C_Under = ybound_min.cwiseMax(0) * xbound_min.cwiseMax(0).transpose()+ybound_max.cwiseMin(0).cwiseAbs() * xbound_max.cwiseMin(0).cwiseAbs().transpose() 
                - (ybound_max.cwiseMax(0) * xbound_min.cwiseMin(0).cwiseAbs().transpose()).cwiseMax(ybound_min.cwiseMin(0).cwiseAbs() * xbound_max.cwiseMax(0).transpose());
                
        C_Over = (ybound_max.cwiseMax(0) * xbound_max.cwiseMax(0).transpose()).cwiseMax(ybound_min.cwiseMin(0).cwiseAbs() * xbound_min.cwiseMin(0).cwiseAbs().transpose()) 
                - ybound_max.cwiseMin(0).cwiseAbs() * xbound_min.cwiseMax(0).transpose()-ybound_min.cwiseMax(0) * xbound_max.cwiseMin(0).cwiseAbs().transpose();
        
        
        

        precomputeHelper(lam);
        
        // std::cout << " passed by precompute " << LX.cols() << '\t' << LY.cols() << std::endl;

    }
    
    void precomputeHelper(double lam){
        const unsigned int n ( LX.rows() );
        const unsigned int  rankx ( LX.cols() );
        const unsigned int  ranky ( LY.cols());
        const unsigned int  m ( rankx*ranky );
       
        Vector p_ex_k = (LY.transpose() * LX).reshaped();
        Vector px_o_py_k = (LY.colwise().mean().transpose() * LX.colwise().mean()).reshaped()*n;

        Xvar =  ( LX.transpose() * LX).diagonal()  ;
        Yvar =  ( LY.transpose() * LY).diagonal() ;

        prob_quadFormMat= (Yvar * Xvar.transpose()).reshaped().array()/static_cast<double> ( n )+n*lam;
        prob_vec = px_o_py_k-p_ex_k;
    }

public:
    const Matrix& xdata;
    const Matrix& ydata;

    Vector h; // the vector of coefficients
    Matrix H; // the matrix of coefficients, h=vec H

    Matrix LX; // the important ones
    Matrix LY; // the important ones

    Vector Xvar; // the sample matrix of the important X ones
    Vector Yvar; // the sample matrix of the important X ones

    Matrix Kxblock; // the kernelfunctions of the important X ones
    Matrix Kyblock; // the kernelfunctions of the important X ones

    Matrix Qx; // the basis transformation matrix
    Matrix Qy; // the basis transformation matrix
    
    Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
    Vector prob_vec; // the important ones
    
    
    Matrix C_Under; // the important ones
    Matrix C_Over; // the important ones
    
    

    KernelMatrix Kx;
    KernelMatrixY Ky;

    LowRank pivx;
    LowRankY pivy;
    
    KernelBasis basx;
    KernelBasisY basy;
    
//  spectral basis bounds. in the first row the mins and in the second row the maxs
    Vector xbound_min;
    Vector xbound_max;
    
    Vector ybound_min;
    Vector ybound_max;
    
//     double tol;
    static constexpr double p = 1.0;
    // Method to print the Kx matrix
    void printKx() const {
        std::cout << "Kx Matrix: " << std::endl;
        std::cout << Kx.full() << std::endl;
    }

    // Method to print the xdata matrix
    void printXdata() const {
        std::cout << "xdata Matrix: " << std::endl;
        std::cout << xdata << std::endl;
    }

    DistributionEmbedding ( const Matrix& xdata_, const Matrix& ydata_) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        Kx ( xdata_ ), // qui ok
        Ky ( ydata_ ),
        basx(Kx,pivx),
        basy(Ky,pivy)
    {
        // std::cout << xdata << std::endl;
        std::cout << "Type of x: " << typeid(xdata).name() << std::endl;
        std::cout << "Type of x_: " << typeid(xdata_).name() << std::endl;
        std::cout << "Type of kx: " << typeid(Kx).name() << std::endl;
        std::cout << "Kx constructor: " << Kx.full() << std::endl;


    }
    const Matrix& getH() const {
        return(H);
    }
    
    unsigned int getSubspaceDimension() const {
        return(LX.cols() + LY.cols());
    }
   
    
    
          /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
        int solveUnconstrained ( double l1, double l2,double prec,double lam )
    {
        
        std::cout << 1 << std::endl;
        precomputeKernelMatrices ( l1, l2,prec,lam );
        std::cout << 2 << std::endl;;
        h = (Qy * ( -prob_vec.cwiseQuotient (prob_quadFormMat).reshaped(Qy.cols(), Qx.cols()) ) * Qx.transpose()).reshaped();
        H = h.reshaped(Qy.cols(), Qx.cols());

        return ( EXIT_SUCCESS );
    }
  
    
    /*
   *    \brief this is the L2 loss function to be evaluated at the validation or test data
   */
    double validationScore(const Matrix& Xs, const Matrix& Ys) const{
        const double n(Xs.cols());
        const Matrix& ky = basy.eval(Ys);
        const Matrix& kx = basx.eval(Xs);
        const Matrix oas = ky * H * kx.transpose();
        
        return(((oas.transpose() * oas).trace()+2.0 * oas.sum())/(n*n)-2.0/n*oas.trace());
    }

    
            /*
   *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
   */
    Matrix condExpfVec ( const Matrix& Xs, bool structuralYes = false ) const
    {

        const Matrix res = Kyblock *H*basx.eval(Xs).transpose() + Matrix::Constant(Kyblock.rows(),Xs.cols(),1);
        if(!structuralYes){
            return(res.array().rowwise()/res.colwise().sum().array());
        }
        // const Matrix resres = res.array().rowwise()/res.colwise().sum().array();

        return ( res.array().cwiseMax(0.0).rowwise()/res.cwiseMax(0.0).colwise().sum().array());
    }
    
            /*
    *    \brief computes conditional expectation of all the rows in Ys
    *    revise this TODO 
    */
    const Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs, bool structuralYes = false  ) const {
        Matrix inter = H * basx.eval(Xs).transpose();
        if(!structuralYes){
            const RowVector norma = Kyblock.colwise().sum() * inter + RowVector::Constant(Xs.cols(),Kyblock.rows());
            const Matrix oida = Ys   * Kyblock; // should be small
            inter.array().rowwise() /= norma.array();
            const Matrix part = Ys.rowwise().sum().replicate(1,Xs.cols()).array().rowwise()/norma.array();
            return ( part+oida * inter );
        }
        const RowVector norma = ((Kyblock * inter) + Matrix::Constant(Kyblock.rows(), Xs.cols(),1)).cwiseMax(0.0).colwise().sum();
            // const Matrix oida = Ys   * Kyblock; // should be small
            inter.array().rowwise() /= norma.array();
            const Matrix part = Ys.rowwise().sum().replicate(1,Xs.cols()).array().rowwise()/norma.array();
            return ( part+ Ys   * (Kyblock *inter).cwiseMax(0.0) );
    }
    
    bool positiveOnGrid() const {
        return(static_cast<bool>((Kyblock * H * Kxblock.transpose()).minCoeff()+1.0 >= 0.0));
    }
    };
} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
