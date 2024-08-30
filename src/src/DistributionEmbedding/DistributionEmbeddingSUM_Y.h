

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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGSUM_Y_H_
#define RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGSUM_Y_H_

// #include "eigen3/Eigen/Dense"
// #include "../../../RRCA/DistributionEmbedding"
// #include <../../home/paultschi/mosek/10.0/tools/platform/linux64x86/h/fusion.h>

namespace RRCA {
namespace DISTRIBUTIONEMBEDDING {

/*
*    \brief tensor product distribution embedding with the x AND y coordinates are sum kernels
*           every coordinate in the x direction gets its own kernel, y gets a sum kernel. The kernel parameters are homogenous across all X and Y kernels
*/
template<typename KernelMatrix, typename LowRank, typename KernelBasis,typename KernelMatrixY = KernelMatrix, typename LowRankY = LowRank, typename KernelBasisY = KernelBasis>
class DistributionEmbeddingSUM_Y {
    typedef typename KernelMatrix::value_type value_type;





public:
    DistributionEmbeddingSUM_Y ( const eigenMatrix& xdata_, const eigenMatrix& ydata_ ) :
        xdata ( xdata_ ),
        xdata_t ( xdata_.rows() ),
        ydata ( ydata_ ),
        h ( xdata_.rows()  ),
        H ( xdata_.rows() ),
        gam ( xdata_.rows()  ),
        LX ( xdata_.rows() ),
        Xvar ( xdata_.rows() ),
        Kxblock ( xdata_.rows() ),
        Qx ( xdata_.rows() ),
        prob_quadFormMat ( xdata_.rows()  ),
        prob_vec ( xdata_.rows()  ),
        Kx ( xdata_.rows() ),
        Ky ( ydata_ ),
        pivx ( xdata_.rows() ),
        basx ( xdata_.rows() ),
        basy ( Ky,pivy ),
        lowerX ( xdata_.rows() ),
        upperX ( xdata_.rows() ),
        midX ( xdata_.rows() ) {
        for ( unsigned int i = 0; i < xdata_.rows(); ++i ) {
            xdata_t[i] = xdata_.row ( i );
            Kx[i]   =  std::make_shared<KernelMatrix> ( xdata_t[i] );
            basx[i] = std::make_shared<KernelBasis> ( * ( Kx[i] ), pivx[i] );
        }

    }
    /*
    *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
    */
    int solve ( double l1, double l2,double prec, double lam,RRCA::Solver solver = RRCA::Solver::Mosek ) {
        precomputeKernelMatrices ( l1,  l2, prec, lam );
        switch ( solver ) {
        case Gurobi:
//             return ( solveGurobi ( lam ) );
            return(EXIT_FAILURE);
        case Mosek:
            return ( solveMosek ( ) );
        }
        return ( EXIT_FAILURE );

    }

#ifdef RRCA_HAVE_MOSEK
//     int solveML ( double l1, double l2,double prec, double lam ) {
//         
//         precomputeKernelMatrices ( l1,  l2, prec );
//         const unsigned int n ( LY[0].rows() );
//         using namespace mosek::fusion;
//         using namespace monty;
//         Model::t M = new Model ( "DistributionEmbeddingTensorSUM_ML" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
//         auto _M = finally ( [&]() {
//             M->dispose();
//         } );
//         //         for the rotated quadratic cone. bounded below by zero
//         Variable::t uu = M->variable ( "uu", xdata.rows() *ydata.rows(), Domain::greaterThan ( 0.0 ) );
//         Variable::t tt = M->variable ( "tt", n, Domain::unbounded() ); // for the exponential cones
//         Variable::t x1 = M->variable ( "x1", n, Domain::greaterThan ( 0.0 ) ); // for the exponential cones
//         std::vector<Variable::t> HH ( xdata.rows()*ydata.rows() );
// 
//         auto summer = new ndarray<Expression::t,1> ( shape ( xdata.rows() *ydata.rows() ) );
// 
//         for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
//             for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
//                 const  int currind ( k*ydata.rows() + l );
//                 const int mx ( LX[k].cols() );
//                 const int my ( LY[l].cols() );
// 
// //      the variables
//                 HH[currind] = M->variable ( new_array_ptr<int,1> ( {mx,my} ), Domain::unbounded() ); // the matrix
// 
// 
// 
// //      compute the expectations of the bases, for the constraints that ensure that it is normalized
//                 Eigen::VectorXd yexp = LY[l].colwise().mean();
//                 Eigen::VectorXd xexp = LX[k].colwise().mean();
//                 const auto yexp_w = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( yexp.data(), shape ( my ) ) );
//                 const auto xexp_w = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( xexp.data(), shape ( mx ) ) );
// 
// //      the normalization
//                 M->constraint ( Expr::dot(xexp_w,Expr::mul ( HH[currind],yexp_w )),Domain::equalsTo ( 0.0 ) );
// 
// //      the setting the tt variables
// //      now be careful for the wrappers, since Eigen is column major
// //      so the below are LY and LX transposed, respectively
//                 const auto LY_w_t = std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( LY[l].data(), shape ( my,n ) ) );
//                 const auto LX_w_t = std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( LX[k].data(), shape ( mx,n ) ) );
//                 Matrix::t LY_t = Matrix::dense ( LY_w_t );
//                 Matrix::t LX_t = Matrix::dense ( LX_w_t );
//                 ( *summer ) [currind] = Expr::mulDiag ( Expr::mul ( LX_t->transpose(), HH[currind] ), LY_t );
// 
//                 Variable::t hh = Var::flatten ( HH[currind] );
//                 M->constraint ( Expr::vstack ( Expr::mul ( 0.5,uu->index ( currind ) ), 1.0, hh ), Domain::inRotatedQCone() );
//             }
//         }
// 
// 
// //      set x1
//         M->constraint ( Expr::sub ( x1,Expr::add ( Expr::constTerm ( n, p ),Expr::add ( std::shared_ptr<ndarray<Expression::t,1>> ( summer ) ) ) ),Domain::equalsTo ( 0.0 ) );
// 
// //      now we check how many data points we have. For n<=10000 let's do the actual problem with exponential cones
//         if ( n<=2000 ) {
// //             this puts all the exponential cones in one shot
//             M->constraint ( Expr::hstack ( x1, Expr::constTerm ( n, 1.0 ), tt ), Domain::inPExpCone() );
//             M->objective ( ObjectiveSense::Maximize, Expr::sub ( Expr::mul ( Expr::sum ( tt ),1.0/static_cast<double> ( n ) ), Expr::mul ( n*lam, Expr::sum ( uu ) ) ) ) ;
//         } else {
// //             M->constraint(tt, Domain::);
//             //             this is the first-order approximation of the exponential cone problem
//             M->objective ( ObjectiveSense::Maximize, Expr::sub ( Expr::mul ( Expr::sum ( x1 ),1.0/static_cast<double> ( n ) ), Expr::mul ( n*lam, Expr::sum ( uu ) ) ) ) ;
//         }
// //      now set in one go all the exponential cones
// 
// 
// 
//         M->solve();
//         if ( M->getPrimalSolutionStatus() == SolutionStatus::Optimal ) {
// //             const auto gammasol = * ( gamma->level() );
//             for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
//                 for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
//                     const unsigned int currind ( k*ydata.rows() + l );
//                     gam ( currind ) = 1.0;
//                     const size_t m ( LX[k].cols() * LY[l].cols() );
//                     auto htsol = * ( HH[currind]->level() );
// //                     Eigen::Map<eigenVector> aux ( &htsol[0], m );
//                     Eigen::Map<eigenMatrix> auxmat ( &htsol[0], LY[l].cols() ,LX[k].cols() );
// //                     h[currind]= Q[currind]*aux;
//                     H[currind] = Qy[l] * auxmat * Qx[k].transpose();
//                     h[currind] = H[currind].reshaped();
//                 }
//             }
//         } else {
//             std::cout << "infeasible  " <<  std::endl;
//             return ( EXIT_FAILURE );
//         }
// 
//         return ( EXIT_SUCCESS );
// 
//     }
#endif

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

//         return ( numer.array().rowwise()/denom.array().unaryExpr([](double arg){return(arg > 1e-8 ? arg : 1e-8);}) );
        return ( numer.array().rowwise()/denom.array());
    }
    
//      /*
// //    *    \brief returns the time series of log(1+g(x_i,y_i)) for the data points in the trainings data set
// //    */
//     eigenVector getLogLikeliTimeSeries(const eigenMatrix& Xs,const eigenMatrix& Ys) const {
//         eigenVector result = eigenVector::Ones ( Xs.cols() );
// 
// 
//         for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
//             for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
//                 eigenMatrix yy = Ys.row(l);
//                 eigenMatrix xx = Xs.row(l);
//                 result += (basy[l]->eval ( yy ) * H[k*ydata.rows() + l] * basx[k]->eval ( xx ).transpose()).diagonal();
//             }
//         }
//         return(result.array().log());
//     }



    const eigenVector& getAlpha() const {
        return{gam};
    }

    void printH() const {
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                std::cout << " alpha " << gam ( k) << std::endl;
                std::cout << h[k].transpose() << std::endl;
        }
    }
    const std::vector<eigenMatrix>& getH() const {
        return ( H );
    }
              /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
    int solveUnconstrained ( double l1, double l2,double prec,double lam )
    {
        precomputeKernelMatrices ( l1, l2,prec,lam );
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                h[k] = (Qy * ( -prob_vec[k].cwiseQuotient (prob_quadFormMat[k]).reshaped(Qy.cols(), Qx[k].cols()) ) * Qx[k].transpose()).reshaped();
                H[k] = h[k].reshaped(Qy.cols(), Qx[k].cols());
        }
        return ( EXIT_SUCCESS );
    }
private:
    const eigenMatrix& xdata;
    std::vector<eigenMatrix> xdata_t;
    const eigenMatrix& ydata;

    std::vector<eigenVector> h; // the vector of coefficients
    std::vector<eigenMatrix> H; // the vector of coefficient matrices
    eigenVector gam; // this is alpha times beta

    std::vector<eigenMatrix> LX; // the important ones
    eigenMatrix LY; // the important ones

    std::vector<eigenVector> Xvar; // the sample matrix of the important X ones
    eigenVector Yvar; // the sample matrix of the important X ones

    std::vector<eigenMatrix> Kxblock; // the kernelfunctions of the important X ones
    eigenMatrix Kyblock; // the kernelfunctions of the important X ones

    std::vector<eigenMatrix> Qx; // the basis transformation matrix
    eigenMatrix Qy; // the basis transformation matrix
//     std::vector<eigenMatrix> Q;

    std::vector<eigenVector> prob_quadFormMat; // this is a vector, because we only need the diagonal
    std::vector<eigenVector> prob_vec; // the important ones

    std::vector<std::shared_ptr<KernelMatrix> > Kx;
    KernelMatrixY Ky;

    std::vector<LowRank>  pivx;
    LowRankY  pivy;

    std::vector<std::shared_ptr<KernelBasis> > basx;
    KernelBasisY basy;

    std::vector<eigenVector> lowerX; //lower bound
    std::vector<eigenVector> upperX; //upper bound
    std::vector<eigenVector> midX; //mean
    
    eigenVector lowerY; //lower bound
    eigenVector upperY; //upper bound
    eigenVector midY; //mean

    double tol;
    static constexpr double p = 1.0;
    static constexpr double ALPHATRESH = 1.0e-07;
//     const double p = 1.0;
//     const double ALPHATRESH = 1.0e-07;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices ( double l1, double l2,double prec, double lam ) {
        tol = prec;
//         std::cout << " start kernel matrices " << std::endl;
            Ky.kernel().l = l2;
            pivy.compute (Ky, tol );
            basy.init ( Ky, pivy.pivots() );
            basy.initSpectralBasisWeights ( pivy );
            Qy =  basy.matrixQ() ;
            Kyblock = basy.eval ( ydata );
            LY = Kyblock * Qy;

        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            Kx[k]->kernel().l = l1;
            pivx[k].compute ( * ( Kx[k] ), tol );
            basx[k]->init ( * ( Kx[k] ), pivx[k].pivots() );
            basx[k]->initSpectralBasisWeights ( pivx[k] );
            Qx[k] =  basx[k]->matrixQ() ;
            Kxblock[k] = basx[k]->eval ( xdata_t[k] );
            LX[k] = Kxblock[k] * Qx[k];
        }
        precomputeHelper(lam);
    }



    void precomputeHelper(double lam) {
        const unsigned int n ( LY.rows() );
        Yvar = ( LY.transpose() * LY ).diagonal() /static_cast<double> ( n ) ;
                        lowerY = LY.colwise().minCoeff();
                upperY = LY.colwise().maxCoeff();
                midY = LY.colwise().mean();

//           for all x coordinates
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                
                Xvar[k] = ( LX[k].transpose() * LX[k] ).diagonal() /static_cast<double> ( n ) ;


                eigenVector p_ex_k = (LY.transpose() * LX[k]).reshaped()/static_cast<double>(n);
                eigenVector px_o_py_k = (LY.colwise().mean().transpose() * LX[k].colwise().mean()).reshaped();

                //      compute the bounds
                lowerX[k] = LX[k].colwise().minCoeff();
                upperX[k] = LX[k].colwise().maxCoeff();
                midX[k] = LX[k].colwise().mean();


                prob_quadFormMat[k] = (Yvar* Xvar[k].transpose()).reshaped().array()+n*lam;
                prob_vec[k] = px_o_py_k-p_ex_k;
        }
    }


    int solveMosek ( ) {
        const unsigned int n ( LY.rows() );
        using namespace mosek::fusion;
        using namespace monty;

        Model::t M = new Model ( "DistributionEmbeddingSUM_Y" );
//             M->setLogHandler ( [=] ( const std::string & msg ) {
//                 std::cout << msg << std::flush;
//             } );
        auto _M = finally ( [&]() {
            M->dispose();
        } );


        //         for the auxiliary quadratic cone for the objective function. bounded below by zero
        Variable::t vv = M->variable ( "vv", xdata.rows() , Domain::greaterThan ( 0.0 ) );
        Variable::t summer = M->variable ( "summer", xdata.rows() , Domain::unbounded() );

        //         the coefficients. these are \tilde h = alpha * h
//         we have one for each coordinate in X
        std::vector<Variable::t> HH ( xdata.rows()  );
        
        eigenVector LYmeans ( LY.colwise().mean() );
        const auto LYmeanswrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( LYmeans.data(), shape ( LYmeans.size() ) ) );
        const auto lowywrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( lowerY.data(), shape ( lowerY.size()) ) ); 
        const auto hiywrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( upperY.data(), shape ( upperY.size()) ) );
        const auto meywrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( midY.data(), shape ( midY.size()) ) );


//      need to work with vectors of shared pointers of gurobi variables

// //      do this for each x coordinate
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                const unsigned int m ( LX[k].cols() * LY.cols() );
                const  int currind ( k );

                HH[currind]   = M->variable ( new_array_ptr<int, 1> ( { ( int ) LX[k].cols(), ( int ) LY.cols() } ), Domain::unbounded() );

                Variable::t ht = Var::flatten ( HH[currind] );

                const auto prob_quadFormMatwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( prob_quadFormMat[currind].data(), shape ( m ) ) );
                const auto quadsqrt = std::make_shared<ndarray<double,1> > ( shape ( m ), [&] ( ptrdiff_t l ) {
                    return sqrt ( ( *prob_quadFormMatwrap ) ( l ) );
                } );
                const auto prob_vecwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( prob_vec[currind].data(), shape ( m ) ) );

                M->constraint ( Expr::vstack ( 0.5,vv->index ( currind ), Expr::mulElm ( quadsqrt,ht ) ), Domain::inRotatedQCone() );
                M->constraint ( Expr::sub ( summer->index ( currind ),Expr::mul ( 2.0,Expr::dot ( prob_vecwrap, ht ) ) ),Domain::equalsTo ( 0.0 ) );

                eigenVector LXmeans ( LX[k].colwise().mean() );

                const auto LXmeanswrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( LXmeans.data(), shape ( LXmeans.size() ) ) );

                

//                 M->constraint ( Expr::mul ( LXmeanswrap,HH[currind] ), Domain::equalsTo ( 0.0 ) );
                M->constraint ( Expr::dot(LXmeanswrap,Expr::mul ( HH[currind],LYmeanswrap )), Domain::equalsTo ( 0.0 ) );
                
//                 const auto lowxwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( lowerX[k].data(), shape ( lowerX[k].size()) ) ); 
//                 const auto hixwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( upperX[k].data(), shape ( upperX[k].size()) ) ); 
//                 const auto mexwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( midX[k].data(), shape ( midX[k].size()) ) ); 
                
                
//                 M->constraint(Expr::dot(lowxwrap,Expr::mul(HH[currind],lowywrap)), Domain::greaterThan(-p));
//                 M->constraint(Expr::dot(lowxwrap,Expr::mul(HH[currind],meywrap)), Domain::greaterThan(-p));
//                 M->constraint(Expr::dot(lowxwrap,Expr::mul(HH[currind],hiywrap)), Domain::greaterThan(-p));
//         
//                 M->constraint(Expr::dot(mexwrap,Expr::mul(HH[currind],lowywrap)), Domain::greaterThan(-p));
//                 M->constraint(Expr::dot(mexwrap,Expr::mul(HH[currind],meywrap)), Domain::greaterThan(-p));
//                 M->constraint(Expr::dot(mexwrap,Expr::mul(HH[currind],hiywrap)), Domain::greaterThan(-p));
//         
//                 M->constraint(Expr::dot(hixwrap,Expr::mul(HH[currind],lowywrap)), Domain::greaterThan(-p));
//                 M->constraint(Expr::dot(hixwrap,Expr::mul(HH[currind],meywrap)), Domain::greaterThan(-p));
//                 M->constraint(Expr::dot(hixwrap,Expr::mul(HH[currind],hiywrap)), Domain::greaterThan(-p));
                
                
//                 Variable::t HH_m = M->variable(new_array_ptr<int, 1>({(int)LX[k].cols(), (int)LY.cols()}), Domain::greaterThan(0.0));
//         Variable::t HH_p = M->variable(new_array_ptr<int, 1>({(int)LX[k].cols(), (int)LY.cols()}), Domain::greaterThan(0.0));
//         const Matrix::t Qxwrap_t = Matrix::dense(std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( Qx[k].data(), shape ( Qx[k].cols(), Qx[k].rows()) ) ));
//         const Matrix::t Qywrap_t = Matrix::dense(std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( Qy.data(), shape ( Qy.cols(), Qy.rows()) ) ));
//         M->constraint(Expr::sub(Expr::mul(Qxwrap_t->transpose(),Expr::mul(HH[k],Qywrap_t)), Expr::sub(HH_p,HH_m)), Domain::equalsTo(0.0));
//         
//         M->constraint ( Expr::sum(HH_m),Domain::lessThan ( p ) );
        }


//         M->objective ( ObjectiveSense::Minimize, Expr::sum ( Expr::add ( Expr::add ( vv,summer ), Expr::mul ( n*lam, uu ) ) ) );
        M->objective ( ObjectiveSense::Minimize, Expr::sum ( Expr::add ( vv,summer ) ) );


        M->solve();
        if ( M->getPrimalSolutionStatus() == SolutionStatus::Optimal && M->getDualSolutionStatus() == SolutionStatus::Optimal) {
//             const auto gammasol = * ( gamma->level() );
            for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                    const unsigned int currind ( k);
                    gam ( currind ) = 1.0;
                    const size_t m ( LX[k].cols() * LY.cols() );
                    const auto htsol = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( * ( HH[currind]->level() ) ) );
                    const Eigen::Map<eigenMatrix> auxmat ( htsol->raw(), LY.cols() ,LX[k].cols() );
                    H[currind] = Qy * auxmat * Qx[k].transpose();
                    h[currind] = H[currind].reshaped();
            }
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }
        return ( EXIT_SUCCESS );
    }
};
} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif // DISTRIBUTIONEMBEDDINGTENSORSUM_H_INCLUDED
