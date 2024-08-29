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
#ifndef RRCA_KERNELREGRESSION_CROSSSECTIONALKERNELREGRESSIONMKL_H_
#define RRCA_KERNELREGRESSION_CROSSSECTIONALKERNELREGRESSIONMKL_H_

// #include "/home/paultschi/mosek/9.3/tools/platform/linux64x86/h/fusion.h"
// #include "eigen3/Eigen/Dense"
// #include "../../../RRCA/KernelRegression"

namespace RRCA
{
namespace KERNELREGRESSION
{



/*
*    \brief This is a single-variable module that considers rank-one kernels of a dictionary of basis functions
*/
template<typename Derived>
class CrossSectionalKernelRegressionRankOneMKL
{
    typedef typename Derived::value_type value_type;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> eigenVector;
    typedef Eigen::Matrix<value_type,  1,Eigen::Dynamic> eigenRowVector;
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> eigenIndexVector;




public:
    CrossSectionalKernelRegressionRankOneMKL ( const eigenMatrix& xdata_, const eigenMatrix& ydata_, const std::vector<std::function<double (double)> > & funs_ ) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        funs(funs_),
        h ( funs.size() ),
        alph ( funs.size() ),
        LX ( funs.size()),
        Kxblock ( funs.size() ),
        Q ( funs.size()),
        Kx ( funs.size() ),
        pivx ( funs.size()),
        basx ( funs.size() )
    {
        for ( unsigned int i = 0; i < funs.size(); ++i ) {
            Kx[i]   =  std::make_shared<RRCA::KernelMatrix<RRCA::RankOneKernel, Derived> > ( xdata );
            Kx[i]->kernel().fun = funs[i];
            basx[i] = std::make_shared<RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::RankOneKernel, Derived> > > ( * ( Kx[i] ), pivx[i] );
        }

    }


    eigenRowVector predict ( const eigenMatrix& Xs ) const
    {
        eigenVector multer = eigenVector::Constant ( Xs.cols(),0.0 );


        for ( unsigned int k = 0; k < funs.size(); ++k ) {
            if ( alph ( k ) > ALPHATRESH ) {
                const eigenMatrix& oida = Xs;
                const eigenMatrix& Kxmultsmall  = basx[k]->eval ( oida );
                multer +=  Kxmultsmall * h[k];
            }
        }

        return ( multer.transpose() );
    }





    const eigenVector& getAlpha() const
    {
        return(alph);
    }
    void printH() const
    {
        for ( unsigned int k = 0; k < funs.size(); ++k ) {
            std::cout << " alpha " << alph ( k ) << std::endl;
            std::cout << h[k].transpose() << std::endl;
        }
    }
    const std::vector<eigenVector>& getH() const
    {
        return ( h );
    }
    
    int solve ( double prec, double lam )
    {
        precomputeKernelMatrices ( prec );
        const unsigned int n ( ydata.cols() );
        const unsigned int m ( funs.size() );
        using namespace mosek::fusion;
        using namespace monty;

        Model::t M = new Model ( "CrossSectionalKernelRegressionRankOneMKL" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
        auto _M = finally ( [&]() {
            M->dispose();
        } );

//          for the optimal convex combination
        Variable::t alpha = M->variable ( "alpha", m, Domain::inRange ( 0.0,1.0 ) );
//         for the rotated quadratic cone. bounded below by zero
        Variable::t uu = M->variable ( "uu", m, Domain::greaterThan ( 0.0 ) );

        //         for the auxiliary quadratic cone for the objective function. bounded below by zero
        Variable::t vv = M->variable ( "vv",  Domain::greaterThan ( 0.0 ) );
        Variable::t summer = M->variable ( "summer", n, Domain::unbounded() );
        
        M->constraint ( Expr::sum(alpha), Domain::equalsTo(1.0) );

        //         the coefficients. these are \tilde h = alpha * h
//         we have one for each coordinate in X
        std::vector<Variable::t> ht ( m );
        
        eigenMatrix nonconsty = ydata;
        
        const auto ywrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( nonconsty.data(), shape (n ) ) );
        
//         Expression::t mySum = Expr::constTerm(ywrap);


        auto prods = std::shared_ptr<ndarray<Expression::t,1> >(new ndarray<Expression::t,1>(shape(m)));
//         (*prods)[0] = Expr::constTerm(ywrap);

// //      do this for each x coordinate
        for ( unsigned int k = 0; k < m; ++k ) {
            ht[k]   = M->variable ( LX[k].cols(), Domain::unbounded());
//             the transposed LX matrices
            const auto LXwrap_t = std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( LX[k].data(), shape ( LX[k].cols(),n ) ) );
            (*prods)[k] = Expr::mul(ht[k],LXwrap_t);

            M->constraint ( Expr::vstack ( Expr::mul ( 0.5,uu->index ( k ) ), alpha->index ( k ), ht[k] ), Domain::inRotatedQCone() ); // the norm
        }
        M->constraint(Expr::sub(summer,Expr::sub(ywrap,Expr::add( std::shared_ptr<ndarray<Expression::t,1> >(prods) ))),Domain::equalsTo(0.0));
        M->constraint ( Expr::vstack ( 0.5,vv,summer ), Domain::inRotatedQCone() );
        M->objective ( ObjectiveSense::Minimize, Expr::sum (Expr::add(  Expr::mul (1.0/static_cast<double>(n),vv), Expr::mul ( n*lam, Expr::sum (uu) )  ) ));


        M->solve();
        if ( M->getPrimalSolutionStatus() == SolutionStatus::Optimal ) {
            const auto alphasol = * ( alpha->level() );
            const auto vvsol = * ( vv->level() );
//             std::cout << "vvsol " << vvsol << std::endl;
            for ( unsigned int k = 0; k < funs.size(); ++k ) {
                alph ( k ) = alphasol[k];
                const size_t m ( LX[k].cols() );
                const auto htsol = * ( ht[k]->level() );
                eigenVector aux ( m );
                for ( size_t i = 0; i < m; ++i ) {
                    aux ( i ) = htsol[i];
                }
                h[k]= Q[k]*aux;
            }
//             std::cout << h.transpose() << std::endl;
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }
        return ( EXIT_SUCCESS );
    }
private:
    const eigenMatrix& xdata;
    const eigenMatrix& ydata;
    
    const std::vector<std::function<double (double)> > & funs;


    std::vector<eigenVector> h; // the vector of coefficients
    eigenVector alph;

    std::vector<eigenMatrix> LX; // the important ones
    std::vector<eigenMatrix> Kxblock; // the kernelfunctions of the important X ones
    std::vector<eigenMatrix> Q;


    std::vector<std::shared_ptr<RRCA::KernelMatrix<RRCA::RankOneKernel, Derived> > > Kx;
    std::vector<RRCA::PivotedCholeskyDecompositon<RRCA::KernelMatrix<RRCA::RankOneKernel, Derived> > >  pivx;
    std::vector<std::shared_ptr<RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::RankOneKernel, Derived> > > > basx;

    double tol;

    static constexpr double ALPHATRESH = 1.0e-07;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices ( double prec )
    {
        tol = prec;
        for ( unsigned int i = 0; i < funs.size(); ++i ) {
//             Kx[i]->kernel().l = l;
//             std::cout << i << std::endl;
            pivx[i].compute ( * ( Kx[i] ), tol );
            basx[i]->init ( * ( Kx[i] ), pivx[i].pivots() );
            basx[i]->initSpectralBasisWeights ( pivx[i] );
            Q[i] =  basx[i]->matrixQ() ;
            Kxblock[i] = basx[i]->eval ( xdata );
            LX[i] = Kxblock[i] * Q[i];
        }
    }



    
};



template<typename Derived>
constexpr double CrossSectionalKernelRegressionRankOneMKL<Derived>::ALPHATRESH;
} // namespace KERNELREGRESSION
}  // namespace RRCA
#endif
