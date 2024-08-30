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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_JOINTDISTRIBUTIONPOLYNOMIALEMBEDDING_SUMX_H_
#define RRCA_DISTRIBUTIONEMBEDDING_JOINTDISTRIBUTIONPOLYNOMIALEMBEDDING_SUMX_H_



namespace RRCA {
namespace DISTRIBUTIONEMBEDDING {

/*
*    \brief specializes joint ditribtion embedding to polynomials. The x dimension is modeled as a sum kernel
*/
template<unsigned int order>
class JointPolynomialDistributionEmbedding_sumx {
public:
    JointPolynomialDistributionEmbedding_sumx ( const Matrix& xdata_, const Matrix& ydata_, bool subSample_ = false ) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        xbasisdim ( binomialCoefficient ( 1+order,1 ) ),
        ybasisdim ( binomialCoefficient ( ydata_.rows()+order,ydata_.rows() ) ),
        xIndex ( 1,order ),
        yIndex ( ydata_.rows(),order ),
        Vx_t ( xdata_.rows() ),
        Vy_t ( ybasisdim,ydata_.cols() ),
        Qx ( xdata_.rows() ),
        xgramL ( xdata_.rows() ),
        subSample ( subSample_ ),
        modelHasBeenSet ( false ) {
//       compute Vandermonde matrices
        const auto &myxSet = xIndex.get_MultiIndexSet();
        for ( const auto &ind1 : myxSet ) {
            xinter.push_back ( ind1 ( 0 ) ); //only one-dimensional power
        }

        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            Vx_t[k].resize ( xbasisdim,xdata_.cols() );
//          compute basisvector
            for ( auto j = 0; j < xinter.size(); ++j ) { // for moments 0 to U.cols()
                Vx_t[k].row ( j )  =  xdata.row ( k ).array().pow ( xinter[j] ).matrix();
            }
            std::cout << " Vx_t k :\n" << Vx_t[k] << std::endl;
        }

        const auto &myySet = yIndex.get_MultiIndexSet();
        for ( const auto &ind1 : myySet ) {
            yinter.push_back ( ind1 );
        }

        for ( unsigned int i = 0; i < ydata.cols(); ++i ) {
//          compute basisvector
            for ( auto j = 0; j < yinter.size(); ++j ) { // for moments 0 to U.cols()
                double accum = 1;
                iVector ind =yinter[j];
                for ( auto k = 0; k < ydata.rows(); ++k ) {
                    accum *= std::pow ( ydata ( k, i ), ind ( k ) );
                }
                Vy_t ( j,i ) = accum;
            }
        }

//      compute the V matrices
        precomputeKernelMatrices();
        // exit(0);

    }
    ~JointPolynomialDistributionEmbedding_sumx() {
        if ( modelHasBeenSet ) {
            M->dispose();
        }
    }
    const Matrix& getH() const {
        return ( h );
    }

    double validationScore ( const Matrix& Xs, const Matrix& Ys ) const {
//         std::vector<Matrix> VVx_t(xdata.rows());
//         Matrix VVy_t(ybasisdim,Ys.cols() );
//         for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
//             VVx_t[k].resize(xbasisdim,Xs.cols());
//             for ( unsigned int i = 0; i < Xs.cols(); ++i ) {
// //          compute basisvector
//                 for ( auto j = 0; j < xinter.size(); ++j ) { // for moments 0 to U.cols()
//                     iVector ind =xinter[j];
//                     VVx_t[k] ( j,i ) = std::pow ( Xs ( k, i ), ind ( k ) );
//                 }
//             }
//         }
//
//         for ( unsigned int i = 0; i < Ys.cols(); ++i ) {
// //          compute basisvector
//             for ( auto j = 0; j < yinter.size(); ++j ) { // for moments 0 to U.cols()
//                 double accum = 1;
//                 iVector ind =yinter[j];
//                 for ( auto k = 0; k < Ys.rows(); ++k ) {
//                     accum *= std::pow ( Ys ( k, i ), ind ( k ) );
//                 }
//                 VVy_t ( j,i ) = accum;
//             }
//         }
//
//
//         const double n(Xs.cols());
//
//         const Matrix oas = VVy_t.transpose() * ygramL * H * xgramL * std::accumulate(VVx_t.begin(),VVx_t.end(),0);
//
//         return((oas.transpose() * oas).trace()/(n*n)-2.0/n*oas.trace());
    }


    /*
    *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
    */
    int solve ( double lam = 0 ) {
#ifdef RRCA_HAVE_MOSEK
        return ( solveMosek ( lam ) );
#endif
        return ( EXIT_FAILURE );
    }




//
    /*
    *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
    */
    int solveUnconstrained ( double lam = 0.0 ) {
        prob_quadFormMat.diagonal().array() +=lam;

        h = ( 2*prob_quadFormMat ).llt().solve ( -prob_vec ).reshaped ( ybasisdim*xbasisdim, xdata.rows() );

        std::cout << " h:\n " << h << std::endl;
        /*
                H = ( -prob_vec.cwiseQuotient ( 2*Vector::Constant(m,1.0+lam) ).reshaped ( Qy.cols(), Qx.cols() ) );
                h = H.reshaped();*/

        return ( EXIT_SUCCESS );
    }


    /*
    *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
    */
    Matrix condExpfVec ( const Matrix& Xs ) const {
//      compute first the new Vandermonde using Xs
//         Matrix newVandermonde_t ( xinter.size(),Xs.cols() );
//         for ( unsigned int i = 0; i < Xs.cols(); ++i ) {
// //          compute basisvector
//             for ( auto j = 0; j < xinter.size(); ++j ) { // for moments 0 to U.cols()
//                 double accum = 1;
//                 RRCA::iVector ind =xinter[j];
//                 for ( auto k = 0; k < Xs.rows(); ++k ) {
//                     accum *= std::pow ( Xs ( k, i ), ind ( k ) );
//                 }
//                 newVandermonde_t ( j,i ) = accum;
//             }
//         }
// //      now compute the kernel matrix block
//
//         const Matrix oas = H * ( xgramL * newVandermonde_t );
//         const Matrix res = Qy * oas;
//         const Matrix resres = res.array().rowwise() /res.colwise().sum().array();
        Matrix resres;
        return ( resres );
    }

    /*
    *    \brief computes conditional expectation of all the rows in Ys
    */
    const Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs ) const {
        if ( subSample ) {
            return ( Ys ( Eigen::all,piv )   * condExpfVec ( Xs ) );
        }
        return ( Ys   * condExpfVec ( Xs ) );
    }


private:
    const Matrix& xdata;
    const Matrix& ydata;
    Matrix xdatasmall;
    Matrix ydatasmall;

    const unsigned int xbasisdim;
    const unsigned int ybasisdim;

    const RRCA::MultiIndexSet<RRCA::iVector> xIndex;
    const RRCA::MultiIndexSet<RRCA::iVector> yIndex;

    std::vector<Matrix> Vx_t; // the important ones
    Matrix Vy_t; // the important ones

    Matrix h; // the matrix of coefficients. for each kernel one
    std::vector<Matrix> H; // the matrix of coefficients, h=vec H

    std::vector<Matrix> Qx; // the basis matrix
    Matrix Qy; // the basis transformation matrix

    Vector prob_vec; // the important ones
    Matrix prob_quadFormMat; // here this needs to be a matrix
    Matrix prob_cholmat; // the Cholesky factor of prob_quadFormMat

    std::vector<Matrix> xgramL;
    Matrix ygramL;


    iVector crossSec;
    const bool subSample;

    std::vector<Index> pivx;
    std::vector<Index> pivy;
    std::vector<Index> piv; // need to select both points

    std::vector<Index> xinter;
    std::vector<RRCA::iVector> yinter;

    bool modelHasBeenSet;
    M_Model M;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices ( ) {
//         compute gram matrix
        if ( subSample ) { // de Marchi trick. apply to untransformed Vandermonde
//             Vector wx = Vx_t.colPivHouseholderQr().solve ( Vector::Ones ( Vx_t.rows() ) );
//             for ( unsigned i = 0; i < Vx_t.cols(); ++i ) {
//                 if ( wx ( i ) > 0.0 || wx ( i ) < 0.0 ) {
//                     pivx.push_back ( i );
//                 }
//             }
//             Vector wy = Vy_t.colPivHouseholderQr().solve ( Vector::Ones ( Vy_t.rows() ) );
//             for ( unsigned i = 0; i < Vy_t.cols(); ++i ) {
//                 if ( wy ( i ) > 0.0 || wy ( i ) < 0.0 ) {
//                     pivy.push_back ( i );
//                 }
//             }
//             std::sort ( pivx.begin(), pivx.end() );
//             std::sort ( pivy.begin(), pivy.end() );
//             std::set_union ( pivx.begin(), pivx.end(), pivy.begin(), pivy.end(), std::back_inserter ( piv ) );
//
//             const double n(piv.size());
//
//
//             xdatasmall = xdata ( Eigen::all, piv );
//             ydatasmall = ydata ( Eigen::all, piv );
//
//             Eigen::SelfAdjointEigenSolver<Matrix> esx ( Vx_t ( Eigen::all, piv ) * Vx_t ( Eigen::all, piv ).transpose()/n );
//             xgramL = esx.operatorInverseSqrt();
//
//             Qx = Vx_t ( Eigen::all, piv ).transpose() * xgramL; // this is Q r
//
//
//             Eigen::SelfAdjointEigenSolver<Matrix> esy ( Vy_t ( Eigen::all, piv ) * Vy_t ( Eigen::all, piv ).transpose()/n );
//             ygramL = esy.operatorInverseSqrt();
//
//             Qy = Vy_t ( Eigen::all, piv ).transpose() * ygramL; // this is Q r

        } else {
            const double n ( Vy_t.cols() );
            for ( Index k = 0; k < xdata.rows(); ++k ) {
                Eigen::SelfAdjointEigenSolver<Matrix> esx ( Vx_t[k] * Vx_t[k].transpose() /n );
                xgramL[k] = esx.operatorInverseSqrt();
                Qx[k] = Vx_t[k].transpose() * xgramL[k];
            }

            // for ( const auto &ind1 : Qx ) {
            //     for ( const auto &ind2 : Qx ) {
            //         // std::cout << " new index " << std::endl;
            //         // std::cout << ind1.transpose() * ind2 << std::endl;
            //     }
            // }


            // this is Q r


            Eigen::SelfAdjointEigenSolver<Matrix> esy ( Vy_t * Vy_t.transpose() /n );
            ygramL = esy.operatorInverseSqrt();

            Qy = Vy_t.transpose() * ygramL; // this is Q r
        }

        precomputeHelper ( );

    }




    void precomputeHelper ( ) {
        const double n ( Qy.rows() );
        Vector p_ex_k ( ybasisdim*xdata.rows() *xbasisdim );
        for ( unsigned int i = 0; i < xdata.rows(); ++i ) {
            p_ex_k.segment ( i*ybasisdim*xbasisdim, ybasisdim*xbasisdim ) = ( Qy.transpose() *Qx[i] ).reshaped();
        }
        prob_vec = -2.0* ( p_ex_k );


        Matrix inter ( xdata.rows() * xbasisdim, xdata.rows() * xbasisdim );
        inter.setZero();
        Matrix interchol;
//      compute the lower block of the covariance matrix
        for ( unsigned int i = 0; i < xdata.rows(); ++i ) {
            inter.block ( i*xbasisdim, i*xbasisdim, xbasisdim, xbasisdim ).diagonal().array() = n;
            for ( unsigned int j = 0; j <xdata.rows(); ++j ) {
                inter.block ( i*xbasisdim, j*xbasisdim, xbasisdim, xbasisdim ) = Qx[i].transpose() * Qx[j];
            }
        }
        inter/=n;
        
        
        PivotedCholeskyDecompositon<Matrix> piv(inter, 1e-14);
        interchol = piv.matrixL();
//      now compute the cholesky factor
        prob_quadFormMat.resize ( inter.rows() *ybasisdim,inter.cols() *ybasisdim );
        prob_quadFormMat.setZero();

        prob_cholmat.resize ( interchol.rows() *ybasisdim,interchol.cols() *ybasisdim );
        prob_cholmat.setZero();
        std::cout << "inter \n" << inter << std::endl;
        std::cout << "interchol \n" << interchol * interchol.transpose() << std::endl;
        // std::cout << "approximation interchol \n" << (inter-interchol * interchol.transpose()).norm() << std::endl;
        // exit(0);


        for ( unsigned int i = 0; i < inter.rows(); ++i ) {
            for ( unsigned int j = 0; j < inter.cols(); ++j ) {
                for ( unsigned int k = 0; k < ybasisdim; ++k ) {
                    prob_quadFormMat ( i*ybasisdim+k,j*ybasisdim+k ) = inter ( i,j );
                }
            }
        }
        for ( unsigned int i = 0; i < interchol.rows(); ++i ) {
            for ( unsigned int j = 0; j < interchol.cols(); ++j ) {
                for ( unsigned int k = 0; k < ybasisdim; ++k ) {
                    prob_cholmat ( i*ybasisdim+k,j*ybasisdim+k ) = interchol ( i,j );
                }
            }
        }
        
        // std::cout << "approximation norm: " << (prob_quadFormMat-prob_cholmat * prob_cholmat.transpose()).norm() << std::endl;
        // std::cout << "approximation\n" << prob_quadFormMat-prob_cholmat * prob_cholmat.transpose() << std::endl;
        // std::cout << "prob_quadFormMat\n" << prob_quadFormMat << std::endl;
        // std::cout << "approximation\n" << prob_cholmat * prob_cholmat.transpose() << std::endl;
        // // exit(0);
    }


#ifdef RRCA_HAVE_MOSEK
    int solveMosek ( double lam ) {
        if ( !modelHasBeenSet ) {
            M = new mosek::fusion::Model ( "JointPolynomialDistributionEmbedding_sumx" );
            // M->setLogHandler ( [=] ( const std::string & msg ) {
            //     std::cout << msg << std::flush;
            // } );


//         contains all coefficient vectors along the row dimension,  because mosek is row major
            M_Variable::t allHH_t = M->variable ( "allH_t", monty::new_array_ptr<int, 1> ( { ( int ) xdata.rows(), ( int ) xbasisdim*ybasisdim } ), M_Domain::unbounded() );
//          for the kernel cones
            M_Variable::t uu = M->variable ( "uu",  M_Domain::greaterThan(0.0) );
//             for the L2 cone
            M_Variable::t vv = M->variable ( "vv",  M_Domain::greaterThan(0.0) );
            
            M_Expression::t linCol = M_Expr::constTerm ( 0.0 );
            auto prob_vecwrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1 ( prob_vec.data(), monty::shape ( xdata.rows()*xbasisdim*ybasisdim ) ) ) ;
            auto lambda = M->parameter ( "lambda" );
            lambda->setValue ( lam );
//          the transpose of the big cholesky matrix
            // std::cout << "prob chol mat\n: " << prob_cholmat << std::endl;
            auto quadcholwrap = std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( prob_cholmat.data(), monty::shape ( prob_cholmat.cols(), prob_cholmat.rows() ) ) );
            M->constraint ( "l2part", M_Expr::vstack ( 0.5, vv, M_Expr::mul(quadcholwrap, M_Expr::flatten(allHH_t) )), M_Domain::inRotatedQCone() );
            M->constraint ( "kernelpart",  M_Expr::vstack ( 0.5, uu,M_Expr::flatten(allHH_t)), M_Domain::inRotatedQCone() );
            
            // std::cout << " so far" << std::endl;
            for ( unsigned int ll = 0; ll < xdata.rows(); ++ll ) {
                M_Variable::t ht = allHH_t->slice ( monty::new_array_ptr<int,1> ( { ( int ) ll , 0} ), monty::new_array_ptr<int,1> ( { ll+1, ( int ) xbasisdim*ybasisdim } ) );
                M_Variable::t HH_t = ht->reshape(xbasisdim, ybasisdim);
                
//      introduce the variable for the PSD constraint
                const M_Matrix::t Gx_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( xgramL[ll].data(), monty::shape ( xgramL[ll].cols(), xgramL[ll].rows() ) ) ) );
                const M_Matrix::t Gy_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( ygramL.data(), monty::shape ( ygramL.cols(), ygramL.rows() ) ) ) );

                M_Variable::t HH_t_bas = M->variable (  monty::new_array_ptr<int, 1> ( { ( int ) xbasisdim, ( int ) ybasisdim } ), M_Domain::unbounded() );

                M->constraint ( M_Expr::sub ( HH_t_bas,  M_Expr::mul ( Gx_wrap, M_Expr::mul ( HH_t,Gy_wrap ) ) ),M_Domain::equalsTo ( 0.0 ) );
                 // quadratic cone for objective function


// // // // // // // // // // // // // // // // // //
// // // // // normalization // // // // // // // //

                Vector LYmeans ( Qy.colwise().mean() );
                Vector LXmeans ( Qx[ll].colwise().mean() );

                auto LYmeanswrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1 ( LYmeans.data(), monty::shape ( LYmeans.size() ) ) );
                auto LXmeanswrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1 ( LXmeans.data(), monty::shape ( LXmeans.size() ) ) );
                linCol = M_Expr::add(linCol, M_Expr::dot ( LXmeanswrap,M_Expr::mul ( HH_t,LYmeanswrap ) ));

// // // // normalization // // // // // // // //
     // positivity constraint

     // embed into polynomial with double order
                const RRCA::MultiIndexSet<RRCA::iVector> myBigIndex ( ( 1+ydata.rows() ),order*2 );
                const RRCA::MultiIndexSet<RRCA::iVector> myBigHalfIndex ( ( 1 +ydata.rows() ),order );


                std::vector<iVector> bigInter;
                const auto &myBigSet = myBigIndex.get_MultiIndexSet();
                for ( const auto &ind1 : myBigSet ) {
                    bigInter.push_back ( ind1 );
                }

                std::vector<iVector> bigHalfInter;
                const auto &myBigHalfSet = myBigHalfIndex.get_MultiIndexSet();
                for ( const auto &ind1 : myBigHalfSet ) {
                    bigHalfInter.push_back ( ind1 );
                }


//      find the map from the coefficient vector to the PSD matrix
                std::map<iVector, std::vector<ijVector>, FMCA_Compare<iVector> > bigIndexMap;
//             fill what's needed for the pos def
                for ( int j = 0; j < bigHalfInter.size(); ++j ) {
                    for ( int i = 0; i <= j; ++i ) {
                        iVector newIndex = bigHalfInter[i] + bigHalfInter[j];
                        ijVector ij;
                        ij << i, j;
                        auto it = bigIndexMap.find ( newIndex );
                        if ( it != bigIndexMap.end() )
                            it->second.push_back ( ij );
                        else
                            bigIndexMap[newIndex].push_back ( ij );
                    }
                }

//          now creat another map  to point to the indices in H_t
//          we know from the bilinear form that every combination is unique
                std::map<iVector, ijVector, FMCA_Compare<iVector> > mapMap;
                for ( int i = 0; i < xinter.size(); ++i ) { //these are the x coordinates
                    for ( int j = 0; j < yinter.size(); ++j ) { //these are the y coordinates
                        RRCA::iVector newIndex ( 1 +ydata.rows() );
                        newIndex <<  yinter[j], xinter[i];
                        RRCA::ijVector ij;
                        ij << i, j; // flipped because H_t is transposed
                        mapMap[newIndex] = ij;
                    }
                }

                //      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
                M_Variable::t MM = M->variable ( RRCA::M_Domain::inPSDCone ( bigHalfInter.size() ) ); // the moment matrix
//      find the coefficients of the big coefficient vector in the matrix of the quadratic form
                const auto &mySet2 = myBigIndex.get_MultiIndexSet();
                for ( const auto &ind : mySet2 ) {
//          find it in the map
                    const std::vector<ijVector> ind1 = bigIndexMap[ind];
                    M_Expression::t ee = M_Expr::constTerm ( 0.0 );
                    for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
                        if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
                            ee = M_Expr::add ( ee,M_Expr::mul ( 2.0,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) ) );
                        } else {// diagonal
                            ee = M_Expr::add ( ee,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) );
                        }
                    }
//          now check whether index is also present in H_t
                    auto it = mapMap.find ( ind );
                    if ( it != mapMap.end() ) { //present, usual semidefinite constraint
                        // std::cout << "setting " << ind.transpose() << "H to i=" << it->second(0 ) << " and j=" << it->second(1) << std::endl;
                        M->constraint ( M_Expr::sub ( ee,HH_t_bas->index ( it->second ( 0 ),it->second ( 1 ) ) ),RRCA::M_Domain::equalsTo ( 0.0 ) );
                    }  else { // corresponding coefficient must be zero
                        // std::cout << "setting " << ind.transpose() << " coefficient must be zero " << std::endl;
                        M->constraint ( ee,M_Domain::equalsTo ( 0.0 ) );
                    }
                }
            }
            M->constraint ( linCol, M_Domain::equalsTo ( 1.0 ) );



            M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::add ( M_Expr::add (M_Expr::mul ( lambda,uu ),vv),M_Expr::dot ( prob_vecwrap, M_Expr::flatten(allHH_t) ) ) );
            modelHasBeenSet = true;
        } else {
            auto lambda = M->getParameter ( "lambda" );
            lambda->setValue ( lam );
        }

        M->solve();

        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal ) {
//             const auto htsol = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( * ( ht->level() ) ) );
            const auto soll = M->getVariable ( "allH_t" );
            M_ndarray_1 htsol   = * ( soll->level() );
            const Eigen::Map<Matrix> auxmat ( htsol.raw(), ybasisdim*xbasisdim,xdata.rows() );
            h = auxmat;
            std::cout << "h:\n" << h << std::endl;
            // h = H.reshaped();
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }
        return ( EXIT_SUCCESS );
    }
#endif

};


} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
