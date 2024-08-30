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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGMKLSUM_Y_H_
#define RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGMKLSUM_Y_H_


// #include "eigen3/Eigen/Dense"
// #include "../../../RRCA/DistributionEmbedding"
// #include <../../home/paultschi/mosek/10.0/tools/platform/linux64x86/h/fusion.h>

namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{

/*
*    \brief tensor product distribution embedding with the x coordinates in multiple kernel learning (MKL)
*           every coordinate in the x dimension gets its own kernel. there is only one kernel for Y (sum kernel)
*/
template<typename KernelMatrix, typename LowRank, typename KernelBasis,typename KernelMatrixY = KernelMatrix, typename LowRankY = LowRank, typename KernelBasisY = KernelBasis>
class DistributionEmbeddingMKLSUM_Y
{
    typedef typename KernelMatrix::value_type value_type;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> eigenVector;
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> eigenIndexVector;




public:
    DistributionEmbeddingMKLSUM_Y ( const eigenMatrix& xdata_, const eigenMatrix& ydata_ ) :
        xdata ( xdata_ ),
        xdata_t ( xdata_.rows() ),
        ydata ( ydata_ ),
        h ( xdata_.rows() ),
        H ( xdata_.rows() ),
        alph ( xdata_.rows() ),
        LX ( xdata_.rows() ),
        Xvar ( xdata_.rows() ),
        Kxblock ( xdata_.rows() ),
        Qx ( xdata_.rows() ),
        prob_quadFormMat ( xdata_.rows() ),
        prob_vec ( xdata_.rows() ),
        Kx ( xdata_.rows() ),
        Ky ( ydata_ ),
        pivx ( xdata_.rows() ),
        basx ( xdata_.rows() ),
        basy ( Ky,pivy )
    {
        for ( unsigned int i = 0; i < xdata_.rows(); ++i ) {
            xdata_t[i] = xdata_.row ( i );
            Kx[i]   =  std::make_shared<KernelMatrix> ( xdata_t[i] );
            basx[i] = std::make_shared<KernelBasis> ( * ( Kx[i] ), pivx[i] );
        }

    }
    /*
    *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
    */
    int solve ( double l1, double l2,double prec, double lam, RRCA::Solver solver = RRCA::Solver::Mosek )
    {
        precomputeKernelMatrices ( l1,  l2, prec );
        switch ( solver ) {
// #ifdef RRCA_HAVE_GUROBI
        case Gurobi:
            return ( EXIT_FAILURE );
//             return ( solveGurobi ( lam ) );
// #endif
#ifdef RRCA_HAVE_MOSEK
        case Mosek:
            return ( solveMosek ( lam ) );
#endif
        default:
            return ( EXIT_FAILURE );
        }
        return ( EXIT_FAILURE );
    }

    /*
    *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
    */
    template<typename l1type, typename l2type>
    int solve ( l1type l1, l2type l2,double prec, double lam, RRCA::Solver solver = RRCA::Solver::Mosek )
    {
        precomputeKernelMatrices ( l1,  l2, prec );
        switch ( solver ) {
        case Gurobi:
            return ( EXIT_FAILURE );
        case Mosek:
            return ( solveMosek ( lam ) );
        }
        return ( EXIT_FAILURE );
    }


       /*
    *    \brief computes conditional expectation in one dimension in Y
    */
    eigenMatrix condExpfY_X ( const std::vector<std::function<double ( const eigenVector& ) > >& funs, const eigenMatrix& Xs ) const {
        const int funsize ( funs.size() );
        eigenMatrix fy = ydata ( Kyblock.rows(),funsize );
        for ( unsigned int k = 0; k < funsize; ++k ) {
            for ( unsigned int l = 0; l < ydata.cols(); ++l ) {
                fy ( l,k ) = funs[k] ( ydata.col ( l ) );
            }
        }
//      now compute a matrix with all the function values

        return ( fy.transpose() * condExpfVec(Xs) );
        
        
    }
    
        /*
    *    \brief computes conditional expectation of all the rows in Ys
    */
    const eigenMatrix condExpfY_X ( const eigenMatrix& Ys, const eigenMatrix& Xs ) const {
        return ( Ys * condExpfVec(Xs) );
    }

    /*
    *    \brief returns n x Xs.cols() matrix that can be multiplied with functino values in each of the n states to return the condition expectation E[f(Y)|X]
    */
    eigenMatrix condExpfVec ( const eigenMatrix& Xs ) const {
        const double n ( Kyblock.rows() );
        eigenMatrix numer = eigenMatrix::Constant ( Kyblock.rows(),Xs.cols(),1 );
        Eigen::RowVectorXd denom = Eigen::RowVectorXd::Constant(Xs.cols(),n);
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                    const eigenMatrix& oida = Xs.row ( k );
                    const eigenMatrix& Kxmultsmall  = basx[k]->eval ( oida ).transpose();
                    numer += Kyblock* H[k] * Kxmultsmall;
                    denom += Kyblock.colwise().sum() * H[k] * Kxmultsmall; 
        }

        return ( numer.array().rowwise()/denom.array() );
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
    const std::vector<eigenMatrix>& getH() const
    {
        return ( H );
    }
private:
    const eigenMatrix& xdata;
    std::vector<eigenMatrix> xdata_t;
    const eigenMatrix& ydata;


    std::vector<eigenVector> h; // the vector of coefficients
    std::vector<eigenMatrix> H; // the vector of coefficient matrices
    eigenVector alph;

    std::vector<eigenMatrix> LX; // the important ones
    eigenMatrix LY; // the important ones
    eigenMatrix LYblock_m; // the important ones

    std::vector<eigenVector> Xvar; // the sample matrix of the important X ones
    eigenVector Yvar; // the sample matrix of the important X ones

    std::vector<eigenMatrix> Kxblock; // the kernelfunctions of the important X ones
    eigenMatrix Kyblock; // the kernelfunctions of the important X ones
//     std::vector<eigenMatrix> Kyblock_m; // the kernelfunctions of the important X ones accrdog to tensor prod rule

    std::vector<eigenMatrix> Qx; // the basis transformation matrix
    eigenMatrix Qy; // the basis transformation matrix

    std::vector<eigenVector> prob_quadFormMat; // this is a vector, because we only need the diagonal
    std::vector<eigenVector> prob_vec; // the important ones

    std::vector<std::shared_ptr<KernelMatrix> > Kx;
    KernelMatrixY Ky;

    std::vector<LowRank>  pivx;
    LowRankY pivy;

    std::vector<std::shared_ptr<KernelBasis> > basx;
    KernelBasisY basy;

    double tol;
    static constexpr double p = 1.0;
    static constexpr double ALPHATRESH = 1.0e-08;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices ( double l1, double l2,double prec )
    {
        tol = prec;
        Ky.kernel().l = l2;

        pivy.compute ( Ky,tol );
        basy.init ( Ky, pivy.pivots() );
        basy.initSpectralBasisWeights ( pivy );

        Qy =  basy.matrixQ() ;
        Kyblock = basy.eval ( ydata );
        LY = Kyblock * Qy;


        for ( unsigned int i = 0; i < xdata.rows(); ++i ) {
            Kx[i]->kernel().l = l1;
            pivx[i].compute ( * ( Kx[i] ), tol );
            basx[i]->init ( * ( Kx[i] ), pivx[i].pivots() );
            basx[i]->initSpectralBasisWeights ( pivx[i] );
            Qx[i] =  basx[i]->matrixQ() ;
            Kxblock[i] = basx[i]->eval ( xdata_t[i] );
            LX[i] = Kxblock[i] * Qx[i];
        }

        precomputeHelper();
    }

    /*
    *    \brief computes the kernel basis and tensorizes it
    *   different parameter for each kernel
    */
    void precomputeKernelMatrices ( const eigenVector& l1, double l2,double prec )
    {
        tol = prec;
        Ky.kernel().l = l2;

        pivy.compute ( Ky,tol );
        basy.init ( Ky, pivy.pivots() );
        basy.initSpectralBasisWeights ( pivy );

        Qy =  basy.matrixQ() ;
        Kyblock = basy.eval ( ydata );
        LY = Kyblock * Qy;

        for ( unsigned int i = 0; i < xdata.rows(); ++i ) {
            Kx[i]->kernel().l = l1 ( i );
            pivx[i].compute ( * ( Kx[i] ), tol );
            basx[i]->init ( * ( Kx[i] ), pivx[i].pivots() );
            basx[i]->initSpectralBasisWeights ( pivx[i] );
            Qx[i] =  basx[i]->matrixQ() ;
            Kxblock[i] = basx[i]->eval ( xdata_t[i] );
            LX[i] = Kxblock[i] * Qx[i];
        }
        precomputeHelper();
    }

    void precomputeHelper()
    {
        const unsigned int n ( LY.rows() );
        Yvar = ( LY.transpose() * LY ).diagonal() /static_cast<double> ( n ) ;
//                         lowerY = LY.colwise().minCoeff();
//                 upperY = LY.colwise().maxCoeff();
//                 midY = LY.colwise().mean();

//           for all x coordinates
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                
                Xvar[k] = ( LX[k].transpose() * LX[k] ).diagonal() /static_cast<double> ( n ) ;


                eigenVector p_ex_k = (LY.transpose() * LX[k]).reshaped()/static_cast<double>(n);
                eigenVector px_o_py_k = (LY.colwise().mean().transpose() * LX[k].colwise().mean()).reshaped();

                //      compute the bounds
//                 lowerX[k] = LX[k].colwise().minCoeff();
//                 upperX[k] = LX[k].colwise().maxCoeff();
//                 midX[k] = LX[k].colwise().mean();


                prob_quadFormMat[k] = (Yvar* Xvar[k].transpose()).reshaped().array();
                prob_vec[k] = px_o_py_k-p_ex_k;
        }
    }

//     int solveGurobi ( double lam )
//     {
//         const unsigned int n ( LY.rows() );
//         std::unique_ptr<GRBEnv> env =  make_unique<GRBEnv>();
// //          env->set(GRB_IntParam_OutputFlag,0);
//         GRBModel model = GRBModel ( *env );
// 
//         GRBQuadExpr qobjvar ( 0 ); // collects the variance part
//         GRBLinExpr lobj ( 0 );
//         GRBQuadExpr qrot ( 0 ); // for the rotated cones
// 
// //      for the optimal convex combination
//         Eigen::VectorXd ubalpha ( Eigen::VectorXd::Constant ( xdata.rows(),1.0 ) );
//         std::unique_ptr<GRBVar[]> alpha ( model.addVars ( NULL, ubalpha.data(), NULL, NULL, NULL,  xdata.rows() ) );
// //         for the rotated quadratic cone. bounded below by zero
//         std::unique_ptr<GRBVar[]> uu ( model.addVars ( NULL, NULL, NULL, NULL, NULL,  xdata.rows() ) );
// 
//         //         the coefficients. these are \tilde h = alpha * h
// //         we have one for each coordinate in X
//         std::vector<std::unique_ptr<GRBVar[]> > ht ( xdata.rows() );
//         std::vector<std::unique_ptr<GRBVar[]> > hp ( xdata.rows() );
//         std::vector<std::unique_ptr<GRBVar[]> > hm ( xdata.rows() );
// 
//         GRBLinExpr alphasum ( 0 );
//         GRBLinExpr usum ( 0 );
// 
//         const eigenVector ysums ( LY.colwise().mean() );
//         const eigenVector LYmeans ( ysums );
//         const unsigned int mY ( LY.cols() );
//         Eigen::DiagonalMatrix<double,Eigen::Dynamic> oidaY ( ( Yvar*n ).cwiseInverse() );
// 
// 
// 
// //      need to work with vectors of shared pointers of gurobi variables
// // //      do this for each x coordinate
//         for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
//             GRBLinExpr lobj1 ( 0 );
//             GRBLinExpr lobj2 ( 0 );
//             GRBLinExpr lobj3 ( 0 );
//             GRBLinExpr lobj4 ( 0 );
//             alphasum += alpha[k];
//             usum += uu[k];
// 
//             qrot = 0;
//             const unsigned int m ( LX[k].cols() * LY.cols() );
// //
//             Eigen::VectorXd lb ( Eigen::VectorXd::Constant ( m,-100.0 ) );
//             Eigen::VectorXd ub ( Eigen::VectorXd::Constant ( m,100.0 ) );
// 
//             ht[k]   = std::unique_ptr<GRBVar[]> ( model.addVars ( lb.data(), ub.data(), NULL, NULL, NULL,  m ) );
//             hp[k]   = std::unique_ptr<GRBVar[]> ( model.addVars ( NULL, ub.data(), NULL, NULL, NULL,  m ) );
//             hm[k]   = std::unique_ptr<GRBVar[]> ( model.addVars ( NULL, ub.data(), NULL, NULL, NULL,  m ) );
// 
//             for ( unsigned int i = 0; i < m; ++i ) {
//                 qobjvar     += prob_quadFormMat[k] ( i ) * ht[k][i] * ht[k][i];
//                 qrot        += ht[k][i] * ht[k][i];
//                 lobj       += 2.0*prob_vec[k] ( i ) * ht[k][i];
//                 model.addConstr ( ht[k][i], GRB_EQUAL, hp[k][i]-hm[k][i] );
//                 lobj2 +=  hp[k][i]*aa[k] ( i )-hm[k][i]*bb[k] ( i );
//             }
//             model.addConstr ( lobj2, GRB_LESS_EQUAL,p );
// 
// 
//             model.addQConstr ( qrot, GRB_LESS_EQUAL,alpha[k] * uu[k] );
// 
//             //         now project onto \psi X
// //      first compute projection matrix
//             const unsigned int mX ( LX[k].cols() );
//             const eigenVector LXmeans ( LX[k].colwise().mean() );
// 
//             for ( unsigned int i = 0; i < mX; ++i ) {
//                 lobj1 = 0;
//                 for ( unsigned int j = 0; j < mY; ++j ) {
//                     lobj1 += ht[k][i*mY + j] * LYmeans ( j );
//                 }
//                 model.addConstr ( lobj1, GRB_EQUAL,0 );
//             }
// 
// 
//             for ( unsigned int i = 0; i < mY; ++i ) {
//                 lobj1 = 0;
//                 for ( unsigned int j = 0; j < mX; ++j ) {
//                     lobj1 += ht[k][i + j*mY] * LXmeans ( j );
//                 }
//                 model.addConstr ( lobj1, GRB_EQUAL,0 );
//             }
//         }
// 
//         model.addConstr ( alphasum, GRB_EQUAL, 1.0 );
// 
// 
//         model.setObjective ( qobjvar + lobj + n * lam * usum );
// 
// 
//         model.optimize();
//         if ( model.get ( GRB_IntAttr_Status ) == GRB_OPTIMAL ) {
// 
//             for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
//                 const size_t m ( LX[k].cols() * LY.cols() );
//                 alph ( k ) = alpha[k].get ( GRB_DoubleAttr_X );
// 
//                 eigenVector aux ( m );
//                 for ( unsigned int i = 0; i < m; ++i ) {
//                     aux ( i ) = ht[k][i].get ( GRB_DoubleAttr_X );
//                 }
//                 h[k]= Q[k]*aux;
//             }
//         } else {
//             std::cout << "infeasible  " <<  std::endl;
//             return ( EXIT_FAILURE );
//         }
// 
//         return ( EXIT_SUCCESS );
//     }

    int solveMosek ( double lam )
    {
        const unsigned int n ( LY.rows() );
        using namespace mosek::fusion;
        using namespace monty;

        Model::t M = new Model ( "DistributionEmbeddingMKLSUM_Y" );
//             M->setLogHandler ( [=] ( const std::string & msg ) {
//                 std::cout << msg << std::flush;
//             } );
        auto _M = finally ( [&]() {
            M->dispose();
        } );

//          for the optimal convex combination
        Variable::t alpha = M->variable ( "alpha", xdata.rows(), Domain::inRange ( 0.0,1.0 ) );
//         for the rotated quadratic cone. bounded below by zero
        Variable::t uu = M->variable ( "uu", xdata.rows(), Domain::greaterThan ( 0.0 ) );

        //         for the auxiliary quadratic cone for the objective function. bounded below by zero
        Variable::t vv = M->variable ( "vv", xdata.rows(), Domain::greaterThan ( 0.0 ) );
        Variable::t summer = M->variable ( "summer", xdata.rows(), Domain::unbounded() );

        //         the coefficients. these are \tilde h = alpha * h
//         we have one for each coordinate in X
        std::vector<Variable::t> HH ( xdata.rows() );
        std::vector<Variable::t> HM ( xdata.rows() );
        std::vector<Variable::t> HP ( xdata.rows() );


        const eigenVector ysums ( LY.colwise().mean() );
        eigenVector LYmeans ( ysums );
        const auto LYmeanswrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( LYmeans.data(), shape ( LYmeans.size() ) ) );
        const unsigned int mY ( LY.cols() );
        Eigen::DiagonalMatrix<double,Eigen::Dynamic> oidaY ( ( Yvar*n ).cwiseInverse() );
        M->constraint ( Expr::sum ( alpha ),Domain::equalsTo ( 1.0 ) );
        const Matrix::t Qywrap_t = Matrix::dense(std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( Qy.data(), shape ( Qy.cols(), Qy.rows()) ) ));


//      need to work with vectors of shared pointers of gurobi variables

// //      do this for each x coordinate
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            const unsigned int m ( LX[k].cols() * LY.cols() );

            HH[k]   = M->variable ( new_array_ptr<int, 1> ( { ( int ) LX[k].cols(), ( int ) LY.cols() } ), Domain::unbounded() );
            HP[k]   = M->variable ( new_array_ptr<int, 1> ( { ( int ) LX[k].cols(), ( int ) LY.cols() } ), Domain::greaterThan ( 0.0 ) );
            HM[k]   = M->variable ( new_array_ptr<int, 1> ( { ( int ) LX[k].cols(), ( int ) LY.cols() } ), Domain::greaterThan ( 0.0 ) );

            Variable::t ht = Var::flatten ( HH[k] );

            const auto prob_quadFormMatwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( prob_quadFormMat[k].data(), shape ( m ) ) );
            const auto quadsqrt = std::make_shared<ndarray<double,1> > ( shape ( m ), [&] ( ptrdiff_t l ) {
                return sqrt ( ( *prob_quadFormMatwrap ) ( l ) );
            } );
            const auto prob_vecwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( prob_vec[k].data(), shape ( m ) ) );
            M->constraint ( Expr::vstack ( 0.5,vv->index ( k ), Expr::mulElm ( quadsqrt,ht ) ), Domain::inRotatedQCone() );
            M->constraint ( Expr::vstack ( Expr::mul ( 0.5,uu->index ( k ) ), alpha->index ( k ), ht ), Domain::inRotatedQCone() );
            M->constraint ( Expr::sub ( summer->index ( k ),Expr::mul ( 2.0,Expr::dot ( prob_vecwrap, ht ) ) ),Domain::equalsTo ( 0.0 ) );

            const Matrix::t Qxwrap_t = Matrix::dense(std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( Qx[k].data(), shape ( Qx[k].cols(), Qx[k].rows()) ) ));
            
            M->constraint(Expr::sub(Expr::mul(Qxwrap_t->transpose(),Expr::mul(HH[k],Qywrap_t)), Expr::sub(HP[k],HM[k])), Domain::equalsTo(0.0));
        
            M->constraint ( Expr::sum(HM[k]),Domain::lessThan ( p ) );
        }


        M->objective ( ObjectiveSense::Minimize, Expr::sum ( Expr::add ( Expr::add ( vv,summer ), Expr::mul ( n*lam, uu ) ) ) );


        M->solve();
        if ( M->getPrimalSolutionStatus() == SolutionStatus::Optimal ) {
            const auto alphasol = * ( alpha->level() );
            for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                    alph ( k ) = alphasol[k];
                    const auto htsol = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( * ( HH[k]->level() ) ) );
                    const Eigen::Map<eigenMatrix> auxmat ( htsol->raw(), LY.cols() ,LX[k].cols() );
                    H[k] = Qy * auxmat * Qx[k].transpose();
                    h[k] = H[k].reshaped();
            }
//             std::cout << h.transpose() << std::endl;
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }
        return ( EXIT_SUCCESS );
    }
};
// template<typename KernelMatrix, typename LowRank, typename KernelBasis>
// constexpr double DistributionEmbeddingMKL<KernelMatrix, LowRank, KernelBasis>::p;
// 
// template<typename KernelMatrix, typename LowRank, typename KernelBasis>
// constexpr double DistributionEmbeddingMKL<KernelMatrix, LowRank, KernelBasis>::ALPHATRESH;
} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
