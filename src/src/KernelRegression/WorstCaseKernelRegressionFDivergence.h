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
#ifndef RRCA_KERNELREGRESSION_WORSTCASEKERNELREGRESSIONFDIVERGENCE_H_
#define RRCA_KERNELREGRESSION_WORSTCASEKERNELREGRESSIONFDIVERGENCE_H_

namespace RRCA
{
namespace KERNELREGRESSION
{

/*
*    \brief minimizes the worst case kernel regression subject to the Q measure being in some entropy vicinity of the empirical measure
*/
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class WorstCaseKernelRegressionFDivergence
{
public:
    WorstCaseKernelRegressionFDivergence ( const RRCA::Matrix& xdata_, const RRCA::Vector& ydata_ , double fdiveps_=1.01) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        fdiveps(fdiveps_),
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
        M->setLogHandler ( [=] ( const std::string & msg ) {
            std::cout << msg << std::flush;
        } );
        auto _M = monty::finally  ( [&]() {
            M->dispose();
        } );
        const unsigned int n = ydata.size();
        const unsigned int m = LX.cols();

//         for the rotated quadratic cone. bounded below by zero
        RRCA::M_Variable::t qq = M->variable ( "qq", n, RRCA::M_Domain::inRange (0.0, 1.0) ); // the probabilities
        M->constraint(RRCA::M_Expr::sum(qq), RRCA::M_Domain::equalsTo(1.0)); // sthe maximum 
        RRCA::M_Variable::t bigmat = M->variable ( "QQ", RRCA::M_Domain::inPSDCone(m+1) ); 

        const RRCA::M_Matrix::t LX_t_wrap = RRCA::M_Matrix::dense(std::make_shared<RRCA::M_ndarray_2>(LX.data(), monty::shape(m, n)));
        const auto ywrap = std::shared_ptr<RRCA::M_ndarray_1> ( new RRCA::M_ndarray_1( ydata.data(), monty::shape (n) ) );
        RRCA::M_Expression::t oida = RRCA::M_Expr::repeat ( RRCA::M_Expr::transpose(qq),m, 0);
        std::cout << "oida " << *oida->getShape() << std::endl;
        RRCA::M_Expression::t  LX_t_diagqq = RRCA::M_Expr::mulElm ( LX_t_wrap, RRCA::M_Expr::repeat ( RRCA::M_Expr::transpose(qq),m, 0) ); // repeat m times in the first dimension

        // set psd matrix
        RRCA::M_Variable::t lowblock = bigmat->slice ( monty::new_array_ptr<int,1> ( { ( int ) 1, ( int ) 1} ), monty::new_array_ptr<int,1> ( { ( int ) (m+1), ( int ) (m + 1)} ) );
        RRCA::M_Variable::t leftcol =  bigmat->slice ( monty::new_array_ptr<int,1> ( { ( int ) 1, ( int ) 0} ), monty::new_array_ptr<int,1> ( { ( int ) (m+1), ( int ) 1} ) );

        M->constraint(RRCA::M_Expr::sub(lowblock,RRCA::M_Expr::add(RRCA::M_Expr::mul(LX_t_diagqq, LX_t_wrap->transpose()),RRCA::M_Matrix::diag(m,lam))), RRCA::M_Domain::equalsTo(0.0));
        M->constraint(RRCA::M_Expr::sub(leftcol,RRCA::M_Expr::mul(LX_t_diagqq, ywrap)), RRCA::M_Domain::equalsTo(0.0));

        // chi square constraint
        M->constraint(RRCA::M_Expr::vstack(0.5, (1.0+fdiveps)/static_cast<double>(n), qq), RRCA::M_Domain::inRotatedQCone()); 

        M->objective ( mosek::fusion::ObjectiveSense::Maximize, RRCA::M_Expr::sub(RRCA::M_Expr::dot(RRCA::M_Expr::mulElm(qq,ywrap),ywrap), bigmat->index(0,0)  ));
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal ) {
            RRCA::M_ndarray_1 htsol = *(lowblock->level());
            const Eigen::Map<RRCA::Matrix> auxmat(htsol.raw(), m,m);
            RRCA::M_ndarray_1 htsol2 = *(qq->level());
            const Eigen::Map<RRCA::Vector> auxmat2(htsol2.raw(), n);
            alpha = auxmat2;
            c = U * (auxmat.llt().solve(LX.transpose() * alpha.asDiagonal() * ydata));
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }
        return ( EXIT_SUCCESS );
    }
private:
    const RRCA::Matrix& xdata;
    const RRCA::Vector& ydata;
    const double fdiveps;


    RRCA::Vector c; // the vector of coefficients
    RRCA::Vector alpha; // the vector of probabilities

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
