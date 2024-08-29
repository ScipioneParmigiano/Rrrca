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
#ifndef RRCA_KERNELREGRESSION_KERNELREGRESSIONMKL_H_
#define RRCA_KERNELREGRESSION_KERNELREGRESSIONMKL_H_

// #include "/home/paultschi/mosek/9.3/tools/platform/linux64x86/h/fusion.h"
// #include "eigen3/Eigen/Dense"
namespace RRCA
{
namespace KERNELREGRESSION
{

/*
*    \brief tensor product distribution embedding with the x coordinates in multiple kernel learning (MKL)
*           every coordinate in the x dimension gets its own kernel
*/
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class KernelRegressionMKL
{
    typedef typename KernelMatrix::value_type value_type;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> eigenVector;
    typedef Eigen::Matrix<value_type,  1,Eigen::Dynamic> eigenRowVector;
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> eigenIndexVector;




public:
    KernelRegressionMKL ( const eigenMatrix& xdata_, const eigenMatrix& ydata_ ) :
        xdata ( xdata_ ),
        xdata_t ( xdata_.rows() ),
        ydata ( ydata_ ),
        h ( xdata_.rows() ),
        alph ( xdata_.rows() ),
        LX ( xdata_.rows() ),
        Kxblock ( xdata_.rows() ),
        Q ( xdata_.rows() ),
        Kx ( xdata_.rows() ),
        pivx ( xdata_.rows() ),
        basx ( xdata_.rows() )
    {
        for ( unsigned int i = 0; i < xdata_.rows(); ++i ) {
            xdata_t[i] = xdata_.row ( i );
            Kx[i]   =  std::make_shared<KernelMatrix> ( xdata_t[i] );
            basx[i] = std::make_shared<KernelBasis> ( * ( Kx[i] ), pivx[i] );
        }

    }


    eigenRowVector predict ( const eigenMatrix& Xs ) const
    {
        eigenVector multer = eigenVector::Constant ( Xs.cols(),0.0 );


        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            if ( alph ( k ) > ALPHATRESH ) {
                const eigenMatrix& oida = Xs.row ( k );
                const eigenMatrix& Kxmultsmall  = basx[k]->eval ( oida );
//                 std::cout << "Kxmultsmall " << Kxmultsmall.rows() << '\t' << Kxmultsmall.cols() << std::endl;
//                 std::cout << "h[k] " << h[k].rows() << '\t' << h[k].cols() << std::endl;
                multer +=  Kxmultsmall * h[k];
            }
        }

        return ( multer.transpose() );
    }





    const eigenVector& getAlpha() const
    {
        return{alph};
    }
    void printH() const
    {
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            std::cout << " alpha " << alph ( k ) << std::endl;
            std::cout << h[k].transpose() << std::endl;
        }
    }
    const std::vector<eigenVector>& getH() const
    {
        return ( h );
    }
    
    int solve ( double l, double prec, double lam )
    {
        precomputeKernelMatrices ( l,prec );
        const unsigned int n ( ydata.cols() );
        const unsigned int m ( xdata.rows() );
        using namespace mosek::fusion;
        using namespace monty;

        Model::t M = new Model ( "KernelRegression" );
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
            for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
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
    std::vector<eigenMatrix> xdata_t;
    const eigenMatrix& ydata;


    std::vector<eigenVector> h; // the vector of coefficients
    eigenVector alph;

    std::vector<eigenMatrix> LX; // the important ones
    std::vector<eigenMatrix> Kxblock; // the kernelfunctions of the important X ones
    std::vector<eigenMatrix> Q;


    std::vector<std::shared_ptr<KernelMatrix> > Kx;
    std::vector<LowRank>  pivx;
    std::vector<std::shared_ptr<KernelBasis> > basx;

    double tol;

    static constexpr double ALPHATRESH = 1.0e-07;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices ( double l, double prec )
    {
        tol = prec;
        for ( unsigned int i = 0; i < xdata.rows(); ++i ) {
            Kx[i]->kernel().l = l;
            pivx[i].compute ( * ( Kx[i] ), tol );
            basx[i]->init ( * ( Kx[i] ), pivx[i].pivots() );
            basx[i]->initSpectralBasisWeights ( pivx[i] );
            Q[i] =  basx[i]->matrixQ() ;
            Kxblock[i] = basx[i]->eval ( xdata_t[i] );
            LX[i] = Kxblock[i] * Q[i];
        }
    }



    
};

/*
*    \brief tensor product distribution embedding with the x coordinates in multiple kernel learning (MKL)
*           every coordinate in the x dimension gets its own kernel
*      partial specialization 
*/
template<typename Derived>
class KernelRegressionRankOneMKL
{
    typedef typename Derived::value_type value_type;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> eigenVector;
    typedef Eigen::Matrix<value_type,  1,Eigen::Dynamic> eigenRowVector;
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> eigenIndexVector;




public:
    KernelRegressionRankOneMKL ( const eigenMatrix& xdata_, const eigenMatrix& ydata_, const std::vector<std::function<double (double)> > & funs ) :
        xdata ( xdata_ ),
        xdata_t ( xdata_.rows() ),
        ydata ( ydata_ ),
        h ( xdata_.rows() ),
        alph ( xdata_.rows() ),
        LX ( xdata_.rows() ),
        Kxblock ( xdata_.rows() ),
        Q ( xdata_.rows() ),
        Kx ( xdata_.rows() ),
        pivx ( xdata_.rows() ),
        basx ( xdata_.rows() )
    {
        for ( unsigned int i = 0; i < xdata_.rows(); ++i ) {
            xdata_t[i] = xdata_.row ( i );
            Kx[i]   =  std::make_shared<RRCA::KernelMatrix<RRCA::RankOneKernel, Derived> > ( xdata_t[i] );
            Kx[i]->kernel().fun = funs[i];
            basx[i] = std::make_shared<RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::RankOneKernel, Derived> > > ( * ( Kx[i] ), pivx[i] );
        }

    }


    eigenRowVector predict ( const eigenMatrix& Xs ) const
    {
        eigenVector multer = eigenVector::Constant ( Xs.cols(),0.0 );


        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            if ( alph ( k ) > ALPHATRESH ) {
                const eigenMatrix& oida = Xs.row ( k );
                const eigenMatrix& Kxmultsmall  = basx[k]->eval ( oida );
//                 std::cout << "Kxmultsmall " << Kxmultsmall.rows() << '\t' << Kxmultsmall.cols() << std::endl;
//                 std::cout << "h[k] " << h[k].rows() << '\t' << h[k].cols() << std::endl;
                multer +=  Kxmultsmall * h[k];
            }
        }

        return ( multer.transpose() );
    }





    const eigenVector& getAlpha() const
    {
        return{alph};
    }
    void printH() const
    {
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
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
        const unsigned int m ( xdata.rows() );
        using namespace mosek::fusion;
        using namespace monty;

        Model::t M = new Model ( "KernelRegressionRankOneMKL" );
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

        //         the coefficients. these are \tilde h = alpha * h
//         we have one for each coordinate in X
        std::vector<Variable::t> ht ( m );
        M->constraint ( Expr::sum(alpha), Domain::equalsTo(1.0) );
        
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
            for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
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
    std::vector<eigenMatrix> xdata_t;
    const eigenMatrix& ydata;


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
        for ( unsigned int i = 0; i < xdata.rows(); ++i ) {
//             Kx[i]->kernel().l = l;
//             std::cout << i << std::endl;
            pivx[i].compute ( * ( Kx[i] ), tol );
            basx[i]->init ( * ( Kx[i] ), pivx[i].pivots() );
            basx[i]->initSpectralBasisWeights ( pivx[i] );
            Q[i] =  basx[i]->matrixQ() ;
            Kxblock[i] = basx[i]->eval ( xdata_t[i] );
            LX[i] = Kxblock[i] * Q[i];
        }
    }



    
};

template<typename KernelMatrix, typename LowRank, typename KernelBasis>
constexpr double KernelRegressionMKL<KernelMatrix, LowRank, KernelBasis>::ALPHATRESH;


template<typename Derived>
constexpr double KernelRegressionRankOneMKL<Derived>::ALPHATRESH;
} // namespace KERNELREGRESSION
}  // namespace RRCA
#endif
