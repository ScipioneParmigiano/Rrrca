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
#ifndef RRCA_KERNELREGRESSION_KERNELLOGISTICREGRESSION_H_
#define RRCA_KERNELREGRESSION_KERNELLOGISTICREGRESSION_H_

namespace RRCA
{
namespace KERNELLOGSTICREGRESSION
{
    
    template <typename T> double sigfn ( T arg )
{
    return ( 1.0/ ( 1.0+exp ( -arg ) ) );
}





// t >= log( 1 + exp(u) ) coordinatewise
void softplus ( M_Model      M,
                M_Expression::t t,
                M_Expression::t u )
{
    int n = ( *t->getShape() ) [0];
    auto z1 = M->variable ( n );
    auto z2 = M->variable ( n );
    M->constraint ( M_Expr::add ( z1, z2 ), M_Domain::equalsTo ( 1 ) );
    M->constraint ( M_Expr::hstack ( z1, M_Expr::constTerm ( n, 1.0 ), M_Expr::sub ( u,t ) ), M_Domain::inPExpCone() );
    M->constraint ( M_Expr::hstack ( z2, M_Expr::constTerm ( n, 1.0 ), M_Expr::neg ( t ) ), M_Domain::inPExpCone() );
}

/*
*    \brief kernel logistic regression. classifier should output -1 for success, and 1 for the complement
*/
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class KernelLogisticRegression
{

public:
    KernelLogisticRegression ( const Matrix& xdata_, const Matrix& ydata_,std::function<double ( double) > classifier_   ) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        classification(ydata_.colwise().sum().unaryExpr(classifier_)),
        Kx ( xdata_ ),
        basx ( Kx,pivx){ 
            // std::cout << "classification:\n" << classification.transpose() << std::endl;
    }


    RowVector predict ( const Matrix& Xs ) const
    {
        const Matrix& Kxmultsmall  = basx.eval(Xs).transpose();
        // std::cout << "exponent:" << h.transpose() * Kxmultsmall << std::endl;

        return((h.transpose() * Kxmultsmall).unaryExpr([](double x) { return sigfn(x); }));
    }
    
    RowVector getExponent ( const Matrix& Xs ) const
    {
        const Matrix& Kxmultsmall  = basx.eval(Xs).transpose();

        return((h.transpose() * Kxmultsmall));
    }

    int solve(double l,  double lam, double prec){
        precomputeKernelMatrix(l,prec);
        
        unsigned int m(LX.cols());
        unsigned int n(LX.rows());
        
        const M_Matrix::t LXwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2  ( LX.data(), monty::shape ( LX.cols(), LX.rows() ) ) ) );

        M_Model M = new mosek::fusion::Model ( "KernelLogisticRegression" );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
        // M->setLogHandler([=](const std::string & msg) {
        //     std::cout << msg << std::flush;
        // } );
        auto quad = M->variable();
        

        M_Variable::t alpha = M->variable ( "q", m, M_Domain::unbounded() ); // these are actually h tilde = B h
        M_Variable::t tt = M->variable ( "tt", n, M_Domain::unbounded() );

        
        M->constraint ( M_Expr::vstack(0.5, quad, alpha), M_Domain::inRotatedQCone());
        M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::add (M_Expr::sum (tt ), M_Expr::mul (n* lam,quad ) ) );
        
         auto signs = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( classification.data(), monty::shape ( n ) ) );
        softplus ( M, tt, M_Expr::mulElm ( M_Expr::mul ( LXwrap->transpose(),alpha ), signs ) );
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal && M->getDualSolutionStatus() == mosek::fusion::SolutionStatus::Optimal ) {
            M_ndarray_1 htsol   = * ( alpha->level() );
            const Eigen::Map<Vector> auxvec ( htsol.raw(), htsol.size() );
            h = Qx * auxvec;
            // std::cout << "auxvec:\n" << auxvec << std::endl;
            // std::cout << "h:\n" << h.transpose() << std::endl;
            // exit(0);
        } else {
            std::cout << "infeasible  " <<  std::endl; 
            return ( EXIT_FAILURE );
        }
        return ( EXIT_SUCCESS );




        return ( EXIT_FAILURE );

    }



   
    void printH() const
    {
        std::cout << h.transpose() << std::endl;
    }
    const Vector& getH() const
    {
        return ( h );
    }
    

private:
    const Matrix& xdata;
    const Matrix& ydata;
    
    Matrix xdatasmall;
    Matrix ydatasmall;
    
     Vector classification; // this vector shows 
     // Vector classificationsmall; 
    


    Vector h; // the vector of coefficients

    Matrix LX; // the important ones
    Matrix Kxblock; // the kernelfunctions of the important X ones
    Matrix Qx;


    KernelMatrix Kx;
    LowRank  pivx;
    KernelBasis basx;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrix ( double l, double prec )
    {

        
//         std::cout << " start precompute " << std::endl;
        Kx.kernel().l = l;

        pivx.compute ( Kx,prec);
        
        basx.init(Kx, pivx.pivots());
            basx.initSpectralBasisWeights(pivx);
            Kxblock = basx.eval(xdata);

        
        
        Qx =  basx.matrixQ() ;
        LX = Kxblock * Qx;
        // std::cout << "Lx:\n" << LX.rows() << '\t' << LX.cols() << std::endl;
//         std::cout << " end precompute " << std::endl;
        
//         precomputeHelper(lam);
    }



    
};




} // namespace KERNELLOGISTICREGRESSION
}  // namespace RRCA
#endif
