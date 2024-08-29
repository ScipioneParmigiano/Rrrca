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
#ifndef RRCA_KERNELREGRESSION_WORSTCASEKERNELREGRESSION_H_
#define RRCA_KERNELREGRESSION_WORSTCASEKERNELREGRESSION_H_


namespace RRCA
{
namespace KERNELREGRESSION
{

/*
*    \brief tensor product distribution embedding with the x coordinates in multiple kernel learning (MKL)
*           every coordinate in the x dimension gets its own kernel
*/
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class WorstCaseKernelRegression
{





public:
    WorstCaseKernelRegression ( const RRCA::Matrix& xdata_, const RRCA::Vector& ydata_ ) :
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

        RRCA::M_Model M = new mosek::fusion::Model ( "WorstCaseKernelRegressionFDivergence" );
        // M->setLogHandler ( [=] ( const std::string & msg ) {
        //     std::cout << msg << std::flush;
        // } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
        const unsigned int n = ydata.size();
        const unsigned int m = LX.cols();
//         for the rotated quadratic cone. bounded below by zero
        RRCA::M_Variable::t uu = M->variable ( "uu", n, RRCA::M_Domain::unbounded () ); // for the q cones
        RRCA::M_Variable::t tt = M->variable ( "tt", RRCA::M_Domain::unbounded () ); // the maximum
        RRCA::M_Variable::t cc = M->variable ( "cc",m, RRCA::M_Domain::unbounded () ); // coefficients
        RRCA::M_Variable::t regger = M->variable ( "reg", RRCA::M_Domain::unbounded () ); // regularization

        const RRCA::M_Matrix::t LX_wrap = RRCA::M_Matrix::dense(std::make_shared<RRCA::M_ndarray_2>(LX.data(), monty::shape(LX.cols(), LX.rows())));
        const auto ywrap = std::shared_ptr<RRCA::M_ndarray_1> ( new RRCA::M_ndarray_1 ( ydata.data(), monty::shape (LX.rows() ) ) );
        RRCA::M_Expression::t forthecone = RRCA::M_Expr::sub (ywrap, RRCA::M_Expr::flatten(RRCA::M_Expr::mul ( cc,LX_wrap) ));

        M->constraint(RRCA::M_Expr::hstack(RRCA::M_Expr::constTerm (n, 0.5 ), uu, forthecone), RRCA::M_Domain::inRotatedQCone()); // sets n rotated cones at a time
        M->constraint(RRCA::M_Expr::sub(mosek::fusion::Var::vrepeat(tt, n),uu), RRCA::M_Domain::greaterThan(0.0)); // sthe maximum 
        M->constraint(RRCA::M_Expr::vstack(0.5, regger, cc), RRCA::M_Domain::inRotatedQCone());


        M->objective ( mosek::fusion::ObjectiveSense::Minimize, RRCA::M_Expr::add(tt, RRCA::M_Expr::mul ( lam, regger)  ));
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal ) {
            RRCA::M_ndarray_1 htsol = *(cc->level());
            const Eigen::Map<RRCA::Vector> auxmat(htsol.raw(), m);
            c = U * auxmat;
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }
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
