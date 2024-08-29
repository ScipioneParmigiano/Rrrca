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
#ifndef RRCA_KERNELREGRESSION_KERNELREGRESSION_H_
#define RRCA_KERNELREGRESSION_KERNELREGRESSION_H_


namespace RRCA
{
namespace KERNELREGRESSION
{

/*
*    \brief tensor product distribution embedding with the x coordinates in multiple kernel learning (MKL)
*           every coordinate in the x dimension gets its own kernel
*/
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class KernelRegression
{





public:
    KernelRegression ( const RRCA::Matrix& xdata_, const RRCA::Vector& ydata_ ) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        K ( xdata_ ),
        bas(K,piv)
    {
        
    }


    RRCA::Vector predict ( const RRCA::Matrix& Xs ) const
    {
        return ( bas.eval ( Xs ) *  c);
    }

    const RRCA::Vector& coefficients() const
    {
        return c;
    } 

    
    int solve ( const std::vector<double>& parms )
    {
        precomputeKernelMatrices ( parms );
        double const lam = parms.back();
        unsigned int m = LX.cols();
        unsigned int n = LX.rows();


        c = U * ( LX.transpose() * LX + n * lam * RRCA::Matrix::Identity(m, m)).llt().solve( LX.transpose() * ydata );
        return ( EXIT_SUCCESS );
    }
private:
    const RRCA::Matrix& xdata;
    const RRCA::Vector& ydata;


    RRCA::Vector c; // the vector of coefficients

    RRCA::Matrix LX; // the important ones
    RRCA::Matrix U; // the important ones

    KernelMatrix K;
    LowRank  piv;
    KernelBasis bas;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices ( const std::vector<double>& parms )
    {
        for (unsigned int i = 0; i < parms.size() - 1; ++i)
        {
            K.kernel().setParameter(parms[i], i);
        }
        piv.compute(K, 1e-4);
        bas.init(K, piv.pivots());
        bas.initNewtonBasisWeights(piv);
        U = bas.matrixU();

        LX = bas.eval(xdata) * U;
    }



    
};


} // namespace KERNELREGRESSION
}  // namespace RRCA
#endif
