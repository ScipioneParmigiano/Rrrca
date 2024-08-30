

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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGTENSORMKL_H_
#define RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDINGTENSORMKL_H_



namespace RRCA {
namespace DISTRIBUTIONEMBEDDING {

/*
*    \brief tensor product distribution embedding with the x AND y coordinates in multiple kernel learning (MKL)
*           every coordinate in the x and y dimension gets its own kernel
*/
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class DistributionEmbeddingTensorMKL {
    typedef typename KernelMatrix::value_type value_type;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> eigenVector;
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> eigenIndexVector;




public:
    DistributionEmbeddingTensorMKL ( const eigenMatrix& xdata_, const eigenMatrix& ydata_ ) :
        xdata ( xdata_ ),
        xdata_t ( xdata_.rows() ),
        ydata ( ydata_ ),
        ydata_t ( ydata_.rows() ),
        h ( xdata_.rows() *ydata_.rows() ),
        H ( xdata_.rows() *ydata_.rows() ),
        gam ( xdata_.rows() *ydata_.rows() ),
        LX ( xdata_.rows() ),
        LY ( ydata_.rows() ),
        Xvar ( xdata_.rows() ),
        Yvar ( ydata_.rows() ),
        Kxblock ( xdata_.rows() ),
        Kyblock ( ydata_.rows() ),
        Qx ( xdata_.rows() ),
        Qy ( ydata_.rows() ),
//         Q ( xdata_.rows() *ydata_.rows() ),
        prob_quadFormMat ( xdata_.rows() *ydata_.rows() ),
        prob_vec ( xdata_.rows() *ydata_.rows() ),
        Kx ( xdata_.rows() ),
        Ky ( ydata_.rows() ),
        pivx ( xdata_.rows() ),
        pivy ( ydata_.rows() ),
        basx ( xdata_.rows() ),
        basy ( ydata_.rows() ),
        aa ( xdata_.rows() * ydata.rows() ),
        bb ( xdata_.rows() * ydata.rows() ) {
        for ( unsigned int i = 0; i < xdata_.rows(); ++i ) {
            xdata_t[i] = xdata_.row ( i );
            Kx[i]   =  std::make_shared<KernelMatrix> ( xdata_t[i] );
            basx[i] = std::make_shared<KernelBasis> ( * ( Kx[i] ), pivx[i] );
//             std::cout << Kx[i]->diagonal() << std::endl;
        }
        for ( unsigned int i = 0; i < ydata_.rows(); ++i ) {
            ydata_t[i] = ydata_.row ( i );
            Ky[i]   =  std::make_shared<KernelMatrix> ( ydata_t[i] );
            basy[i] = std::make_shared<KernelBasis> ( * ( Ky[i] ), pivy[i] );
        }
    }
    /*
    *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
    */
    int solve ( double l1, double l2,double prec, double lam,RRCA::Solver solver = RRCA::Solver::Gurobi ) {
        precomputeKernelMatrices ( l1,  l2, prec );
        switch ( solver ) {
        case Gurobi:
            return ( solveGurobi ( lam ) );
        case Mosek:
            return ( solveMosek ( lam ) );
        }
        return ( EXIT_FAILURE );

    }

#ifdef RRCA_HAVE_MOSEK
    int solveML ( double l1, double l2,double prec, double lam ) {
        
        precomputeKernelMatrices ( l1,  l2, prec );
        const unsigned int n ( LY[0].rows() );
        using namespace mosek::fusion;
        using namespace monty;
        Model::t M = new Model ( "DistributionEmbeddingTensorMKL_ML" );
        M->setLogHandler ( [=] ( const std::string & msg ) {
            std::cout << msg << std::flush;
        } );
        auto _M = finally ( [&]() {
            M->dispose();
        } );
        //          for the optimal convex combination
        Variable::t gamma = M->variable ( "gamma", xdata.rows() *ydata.rows(), Domain::inRange ( 0.0,1.0 ) );
        //         for the rotated quadratic cone. bounded below by zero
        Variable::t uu = M->variable ( "uu", xdata.rows() *ydata.rows(), Domain::greaterThan ( 0.0 ) );
        Variable::t tt = M->variable ( "tt", n, Domain::unbounded() ); // for the exponential cones
        Variable::t x1 = M->variable ( "x1", n, Domain::greaterThan ( 0.0 ) ); // for the exponential cones
        std::vector<Variable::t> HH ( xdata.rows()*ydata.rows() );
//         Variable::t total = M->variable ( "total", Domain::greaterThan ( 0.0 ) );


        auto summer = new ndarray<Expression::t,1> ( shape ( xdata.rows() *ydata.rows() ) );
        M->constraint ( Expr::sum ( gamma ),Domain::equalsTo ( 1.0 ) );


        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                const  int currind ( k*ydata.rows() + l );
                const int mx ( LX[k].cols() );
                const int my ( LY[l].cols() );

//      the variables
                HH[currind] = M->variable ( new_array_ptr<int,1> ( {mx,my} ), Domain::unbounded() ); // the matrix



//      compute the expectations of the bases, for the constraints that ensure that it is normalized
                Eigen::VectorXd yexp = LY[l].colwise().mean();
                Eigen::VectorXd xexp = LX[k].colwise().mean();
                const auto yexp_w = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( yexp.data(), shape ( my ) ) );
                const auto xexp_w = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( xexp.data(), shape ( mx ) ) );

//      the normalization
                M->constraint ( Expr::mul ( xexp_w,HH[currind] ),Domain::equalsTo ( 0.0 ) );
                M->constraint ( Expr::mul ( HH[currind],yexp_w ),Domain::equalsTo ( 0.0 ) );




//      the setting the tt variables
//      now be careful for the wrappers, since Eigen is column major
//      so the below are LY and LX transposed, respectively
                const auto LY_w_t = std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( LY[l].data(), shape ( my,n ) ) );
                const auto LX_w_t = std::shared_ptr<ndarray<double, 2> > ( new ndarray<double, 2> ( LX[k].data(), shape ( mx,n ) ) );
                Matrix::t LY_t = Matrix::dense ( LY_w_t );
                Matrix::t LX_t = Matrix::dense ( LX_w_t );
                ( *summer ) [currind] = Expr::mulDiag ( Expr::mul ( LX_t->transpose(), HH[currind] ), LY_t );

                Variable::t hh = Var::flatten ( HH[currind] );
                M->constraint ( Expr::vstack ( Expr::mul ( 0.5,uu->index ( currind ) ), gamma->index ( currind ), hh ), Domain::inRotatedQCone() );
            }
        }


//      set x1
        M->constraint ( Expr::sub ( x1,Expr::add ( Expr::constTerm ( n, p ),Expr::add ( std::shared_ptr<ndarray<Expression::t,1>> ( summer ) ) ) ),Domain::equalsTo ( 0.0 ) );

//      now we check how many data points we have. For n<=10000 let's do the actual problem with exponential cones
        if ( n<=2000 ) {
//             this puts all the exponential cones in one shot
            M->constraint ( Expr::hstack ( x1, Expr::constTerm ( n, 1.0 ), tt ), Domain::inPExpCone() );
            M->objective ( ObjectiveSense::Maximize, Expr::sub ( Expr::mul ( Expr::sum ( tt ),1.0/static_cast<double> ( n ) ), Expr::mul ( n*lam, Expr::sum ( uu ) ) ) ) ;
        } else {
//             M->constraint(tt, Domain::);
            //             this is the first-order approximation of the exponential cone problem
            M->objective ( ObjectiveSense::Maximize, Expr::sub ( Expr::mul ( Expr::sum ( x1 ),1.0/static_cast<double> ( n ) ), Expr::mul ( n*lam, Expr::sum ( uu ) ) ) ) ;
        }
//      now set in one go all the exponential cones



        M->solve();
        if ( M->getPrimalSolutionStatus() == SolutionStatus::Optimal ) {
            const auto gammasol = * ( gamma->level() );
            for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                    const unsigned int currind ( k*ydata.rows() + l );
                    gam ( currind ) = gammasol[currind];
                    const size_t m ( LX[k].cols() * LY[l].cols() );
                    auto htsol = * ( HH[currind]->level() );
                    Eigen::Map<eigenVector> aux ( &htsol[0], m );
                    Eigen::Map<eigenMatrix> auxmat ( &htsol[0], LY[l].cols() ,LX[k].cols() );
//                     h[currind]= Q[currind]*aux;
                    H[currind] = Qy[l] * auxmat * Qx[k].transpose();
                    h[currind] = Eigen::Map<eigenVector>(H[currind].data(), H[currind].rows()*H[currind].cols());
                }
            }
//             std::cout << h.transpose() << std::endl;
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }

        return ( EXIT_SUCCESS );

    }
#endif




//        /*
//    *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
//    */
//     int solve(const eigenVector& l1, const eigenVector& l2,double prec, double lam, RRCA::Solver solver = RRCA::Solver::Gurobi){
//         precomputeKernelMatrices( l1,  l2, prec);
//         switch(solver){
//             case Gurobi:
//                 return(solveGurobi(lam));
//             case Mosek:
//                 return(solveMosek(lam));
//         }
//         return(EXIT_FAILURE);
//     }

    /*
    *    \brief computes conditional expectation in one dimension in Y
    */
    eigenMatrix condExpfY_X ( const std::vector<std::function<double ( const eigenVector& ) > >& funs, const eigenMatrix& Xs ) const {
        const double n ( Kyblock[0].rows() );
        eigenMatrix multer = eigenMatrix::Constant ( Kyblock[0].rows(),Xs.cols(),1.0/n );


        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                if ( gam ( k*ydata.rows() + l ) > ALPHATRESH ) {
                    const eigenMatrix& oida = Xs.row ( k );
                    const eigenMatrix& Kxmultsmall  = basx[k]->eval ( oida ).transpose();
                    multer += Kyblock[l] * H[k*ydata.rows() + l] * Kxmultsmall /n;
                }
            }
        }

        const int funsize ( funs.size() );
        eigenMatrix fy = ydata ( Kyblock[0].rows(),funsize );
        for ( unsigned int k = 0; k < funsize; ++k ) {
            for ( unsigned int l = 0; l < ydata.cols(); ++l ) {
                fy ( l,k ) = funs[k] ( ydata.col ( l ) );
            }
        }
//      now compute a matrix with all the function values

        return ( fy.transpose() * multer );
    }

    eigenMatrix condExpfVec ( const eigenMatrix& Xs ) const {
        const double n ( Kyblock[0].rows() );
        eigenMatrix multer = eigenMatrix::Constant ( Kyblock[0].rows(),Xs.cols(),1.0/n );


        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                if ( gam ( k*ydata.rows() + l ) > ALPHATRESH ) {
                                        const eigenMatrix& oida = Xs.row ( k );
                    const eigenMatrix& Kxmultsmall  = basx[k]->eval ( oida ).transpose();
                    multer += Kyblock[l] * H[k*ydata.rows() + l] * Kxmultsmall /n;
                }
            }
        }

        return ( multer );
    }
    
     /*
//    *    \brief returns the time series of log(1+g(x_i,y_i)) for the data points in the trainings data set
//    */
    eigenVector getLogLikeliTimeSeries(const eigenMatrix& Xs,const eigenMatrix& Ys) const {
        eigenVector result = eigenVector::Ones ( Xs.cols() );
//         const eigenMatrix Xs_t = Xs.transpose();
//         const eigenMatrix Ys_t = Ys.transpose();


        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                eigenMatrix yy = Ys.row(l);
                eigenMatrix xx = Xs.row(l);
                result += (basy[l]->eval ( yy ) * H[k*ydata.rows() + l] * basx[k]->eval ( xx ).transpose()).diagonal();
            }
        }
        return(result.array().log());
    }



    const eigenVector& getAlpha() const {
        return{gam};
    }

    void printH() const {
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                std::cout << " alpha " << gam ( k*ydata.rows() + l ) << std::endl;
                std::cout << h[k*ydata.rows() + l].transpose() << std::endl;
            }
        }
    }
    const std::vector<eigenVector>& getH() const {
        return ( h );
    }
private:
    const eigenMatrix& xdata;
    std::vector<eigenMatrix> xdata_t;
    const eigenMatrix& ydata;
    std::vector<eigenMatrix> ydata_t;

    std::vector<eigenVector> h; // the vector of coefficients
    std::vector<eigenMatrix> H; // the vector of coefficient matrices
    eigenVector gam; // this is alpha times beta

    std::vector<eigenMatrix> LX; // the important ones
    std::vector<eigenMatrix> LY; // the important ones

    std::vector<eigenVector> Xvar; // the sample matrix of the important X ones
    std::vector<eigenVector> Yvar; // the sample matrix of the important X ones

    std::vector<eigenMatrix> Kxblock; // the kernelfunctions of the important X ones
    std::vector<eigenMatrix> Kyblock; // the kernelfunctions of the important X ones

    std::vector<eigenMatrix> Qx; // the basis transformation matrix
    std::vector<eigenMatrix> Qy; // the basis transformation matrix
//     std::vector<eigenMatrix> Q;

    std::vector<eigenVector> prob_quadFormMat; // this is a vector, because we only need the diagonal
    std::vector<eigenVector> prob_vec; // the important ones

    std::vector<std::shared_ptr<KernelMatrix> > Kx;
    std::vector<std::shared_ptr<KernelMatrix> > Ky;

    std::vector<LowRank>  pivx;
    std::vector<LowRank>  pivy;

    std::vector<std::shared_ptr<KernelBasis> > basx;
    std::vector<std::shared_ptr<KernelBasis> > basy;

    std::vector<eigenVector> aa;
    std::vector<eigenVector> bb;

    double tol;
    static constexpr double p = 1.0;
    static constexpr double ALPHATRESH = 1.0e-07;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices ( double l1, double l2,double prec ) {
        tol = prec;

        for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
            Ky[l]->kernel().l = l2;
            pivy[l].compute ( * ( Ky[l] ), tol );
            basy[l]->init ( * ( Ky[l] ), pivy[l].pivots() );
            basy[l]->initSpectralBasisWeights ( pivy[l] );
            Qy[l] =  basy[l]->matrixQ() ;
            Kyblock[l] = basy[l]->eval ( ydata_t[l] );
            LY[l] = Kyblock[l] * Qy[l];
        }

        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            Kx[k]->kernel().l = l1;
            pivx[k].compute ( * ( Kx[k] ), tol );
            basx[k]->init ( * ( Kx[k] ), pivx[k].pivots() );
            basx[k]->initSpectralBasisWeights ( pivx[k] );
            Qx[k] =  basx[k]->matrixQ() ;
            Kxblock[k] = basx[k]->eval ( xdata_t[k] );
            LX[k] = Kxblock[k] * Qx[k];
        }

        precomputeHelper();
    }



    void precomputeHelper() {
        const unsigned int n ( LY[0].rows() );


//           for all x coordinates
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
//                 //           compute big Q matrix
//                 Q[k*ydata.rows() + l].resize ( LX[k].cols() *LY[l].cols(),LX[k].cols() *LY[l].cols() );
//                 for ( unsigned  int i = 0; i < Qx[k].cols(); i++ ) {
//                     for ( unsigned int j = 0; j < Qx[k].rows(); j++ ) {
//                         Q[k*ydata.rows() + l].block ( j*Qy[l].rows(), i*Qy[l].cols(), Qy[l].rows(), Qy[l].cols() ) =  Qx[k] ( j,i ) *Qy[l];
//                     }
//                 }


                // compute the P expectation of the subspace and compute the Px \otimes Py matrix
                eigenVector p_ex_k = (LY[l].transpose() * LX[k]).reshaped()/static_cast<double>(n);
                eigenVector px_o_py_k = (LY[l].colwise().mean().transpose() * LX[k].colwise().mean()).reshaped();

//                 eigenVector LYsums ( p * LY[l].colwise().mean() );
//                 Yvar[l].resize ( LY[l].cols() );
                Yvar[l] = ( LY[l].transpose() * LY[l] ).diagonal() /static_cast<double> ( n ) ;

//                 Xvar[k].resize ( LX[k].cols() );
                Xvar[k] = ( LX[k].transpose() * LX[k] ).diagonal() /static_cast<double> ( n ) ;

/*
                for ( auto i = 0; i < LX[k].cols(); ++i ) {
                    for ( auto j = 0; j < LY[l].cols(); ++j ) {
                        p_ex_k ( i*LY[l].cols() + j ) = ( LX[k].col ( i ).cwiseProduct ( LY[l].col ( j ) ) ).mean();
                        px_o_py_k ( i*LY[l].cols() + j ) = LX[k].col ( i ).mean() * LYsums ( j ) ;
                    }
                }*/

                //      compute the bounds
                eigenVector boundXlo = LX[k].colwise().minCoeff();
                eigenVector boundXhi = LX[k].colwise().maxCoeff();
                eigenVector boundYlo = LY[l].colwise().minCoeff();
                eigenVector boundYhi = LY[l].colwise().maxCoeff();
                unsigned int m ( LY[l].cols() *LX[k].cols() );



                eigenMatrix boundMat ( m,4 );

                boundMat <<  Eigen::Map<Eigen::VectorXd> ( Eigen::MatrixXd ( boundYlo * boundXlo.transpose() ).data(),m ),
                         Eigen::Map<Eigen::VectorXd> ( Eigen::MatrixXd ( boundYhi * boundXlo.transpose() ).data(),m ),
                         Eigen::Map<Eigen::VectorXd> ( Eigen::MatrixXd ( boundYlo * boundXhi.transpose() ).data(),m ),
                         Eigen::Map<Eigen::VectorXd> ( Eigen::MatrixXd ( boundYhi * boundXhi.transpose() ).data(),m );

                aa[k*ydata.rows() + l] = boundMat.rowwise().minCoeff();
                bb[k*ydata.rows() + l ] = boundMat.rowwise().maxCoeff();



/*
                prob_quadFormMat[k*ydata.rows() + l].resize ( LX[k].cols() * LY[l].cols() );

//          now compute kernel matrix
                for ( int i = 0; i < LX[k].cols(); i++ ) {
                    for ( int j = 0; j < LY[l].cols(); j++ ) {
                        prob_quadFormMat[k*ydata.rows() + l] ( i*LY[l].cols()+j ) = Xvar[k] ( i ) *Yvar[l] ( j ) ;
                    }
                }*/
                prob_quadFormMat[k*ydata.rows() + l] = (Yvar[l] * Xvar[k].transpose()).reshaped();
                prob_vec[k*ydata.rows() + l] = px_o_py_k-p_ex_k;
            }
        }
    }

    int solveGurobi ( double lam ) {
        const unsigned int n ( LY[0].rows() );
        std::unique_ptr<GRBEnv> env =  make_unique<GRBEnv>();
        GRBModel model = GRBModel ( *env );

        GRBQuadExpr qobjvar ( 0 ); // collects the variance part
        GRBLinExpr lobj ( 0 );
        GRBQuadExpr qrot ( 0 ); // for the rotated cones

//      for the optimal convex combination
        const Eigen::VectorXd ubalpha ( Eigen::VectorXd::Constant ( xdata.rows() * ydata.rows(),1.0 ) );
        std::unique_ptr<GRBVar[]> alpha ( model.addVars ( NULL, ubalpha.data(), NULL, NULL, NULL,  xdata.rows() * ydata.rows() ) );
//         for the rotated quadratic cone. bounded below by zero
        std::unique_ptr<GRBVar[]> uu ( model.addVars ( NULL, NULL, NULL, NULL, NULL,  xdata.rows()  * ydata.rows() ) );

        //         the coefficients. these are \tilde h = alpha * h
//         we have one for each coordinate in X
        std::vector<std::unique_ptr<GRBVar[]> > ht ( xdata.rows() * ydata.rows() );
        std::vector<std::unique_ptr<GRBVar[]> > hp ( xdata.rows() * ydata.rows() );
        std::vector<std::unique_ptr<GRBVar[]> > hm ( xdata.rows() * ydata.rows() );

        GRBLinExpr alphasum ( 0 );
        GRBLinExpr usum ( 0 );

//      need to work with vectors of shared pointers of gurobi variables
// //      do this for each x coordinate
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                const unsigned int currind ( k*ydata.rows() + l );

                GRBLinExpr lobj1 ( 0 );
                GRBLinExpr lobj2 ( 0 );
                GRBLinExpr lobj3 ( 0 );
                GRBLinExpr lobj4 ( 0 );
                alphasum += alpha[currind];
                usum += uu[currind];

                qrot = 0;
                const unsigned int m ( LX[k].cols() * LY[l].cols() );
//
                Eigen::VectorXd lb ( Eigen::VectorXd::Constant ( m,-100.0 ) );
                Eigen::VectorXd ub ( Eigen::VectorXd::Constant ( m,100.0 ) );

                ht[currind]   = std::unique_ptr<GRBVar[]> ( model.addVars ( lb.data(), ub.data(), NULL, NULL, NULL,  m ) );
                hm[currind]   = std::unique_ptr<GRBVar[]> ( model.addVars ( NULL, ub.data(), NULL, NULL, NULL,  m ) );
                hp[currind]   = std::unique_ptr<GRBVar[]> ( model.addVars ( NULL, ub.data(), NULL, NULL, NULL,  m ) );

                for ( unsigned int i = 0; i < m; ++i ) {
                    qobjvar     += prob_quadFormMat[currind] ( i ) * ht[currind][i] * ht[currind][i];
                    qrot        += ht[currind][i] * ht[currind][i];
                    lobj       += 2.0*prob_vec[currind] ( i ) * ht[currind][i];
                    model.addConstr ( ht[currind][i], GRB_EQUAL, hp[currind][i]-hm[currind][i] );
                    lobj2 +=  hp[currind][i]*aa[currind] ( i )-hm[currind][i]*bb[currind] ( i );
                }
                model.addConstr ( lobj2, GRB_LESS_EQUAL,p );
                model.addQConstr ( qrot, GRB_LESS_EQUAL,alpha[currind] * uu[currind] );

                //         now project onto \psi X
//      first compute projection matrix
                const unsigned int mX ( LX[k].cols() );
                const eigenVector LXmeans ( LX[k].colwise().mean() );


                const eigenVector ysums ( LY[l].colwise().mean() );
                const eigenVector LYmeans ( ysums );
                const unsigned int mY ( LY[l].cols() );
//                 Eigen::DiagonalMatrix<double,Eigen::Dynamic> oidaY ( ( Yvar[l]*n ).cwiseInverse() );

                for ( unsigned int i = 0; i < mX; ++i ) {
                    lobj1 = 0;
                    for ( unsigned int j = 0; j < mY; ++j ) {
                        lobj1 += ht[currind][i*mY + j] * LYmeans ( j );
                    }
                    model.addConstr ( lobj1, GRB_EQUAL,0 );
                }


                for ( unsigned int i = 0; i < mY; ++i ) {
                    lobj1 = 0;
                    for ( unsigned int j = 0; j < mX; ++j ) {
                        lobj1 += ht[currind][i + j*mY] * LXmeans ( j );
                    }
                    model.addConstr ( lobj1, GRB_EQUAL,0 );
                }
            }
        }




        model.addConstr ( alphasum, GRB_EQUAL, 1.0 );
        model.setObjective ( qobjvar + lobj + n * lam * usum );
        model.optimize();
        if ( model.get ( GRB_IntAttr_Status ) == GRB_OPTIMAL ) {
            for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                    const size_t m ( LX[k].cols() * LY[l].cols() );
                    const unsigned int currind ( k*ydata.rows() + l );
                    gam ( currind ) = alpha[currind].get ( GRB_DoubleAttr_X );

                    eigenVector aux ( m );
                    for ( unsigned int i = 0; i < m; ++i ) {
                        aux ( i ) = ht[currind][i].get ( GRB_DoubleAttr_X );
                    }
//                     h[currind]= Q[currind]*aux;
                    Eigen::Map<eigenMatrix> auxmat ( aux.data(), LY[l].cols() ,LX[k].cols() );
                    H[currind] = Qy[l] * auxmat * Qx[k].transpose();
                    h[currind] = Eigen::Map<eigenVector>(H[currind].data(), H[currind].rows()*H[currind].cols());
                }
            }
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }



        return ( EXIT_SUCCESS );
    }

    int solveMosek ( double lam ) {
        const unsigned int n ( LY[0].rows() );
        using namespace mosek::fusion;
        using namespace monty;

        Model::t M = new Model ( "DistributionEmbeddingTensorMKL" );
//             M->setLogHandler ( [=] ( const std::string & msg ) {
//                 std::cout << msg << std::flush;
//             } );
        auto _M = finally ( [&]() {
            M->dispose();
        } );

//          for the optimal convex combination
        Variable::t gamma = M->variable ( "gamma", xdata.rows() *ydata.rows(), Domain::inRange ( 0.0,1.0 ) );
//         for the rotated quadratic cone. bounded below by zero
        Variable::t uu = M->variable ( "uu", xdata.rows() *ydata.rows(), Domain::greaterThan ( 0.0 ) );

        //         for the auxiliary quadratic cone for the objective function. bounded below by zero
        Variable::t vv = M->variable ( "vv", xdata.rows() *ydata.rows(), Domain::greaterThan ( 0.0 ) );
        Variable::t summer = M->variable ( "summer", xdata.rows() *ydata.rows(), Domain::unbounded() );

        //         the coefficients. these are \tilde h = alpha * h
//         we have one for each coordinate in X
        std::vector<Variable::t> HH ( xdata.rows() *ydata.rows() );
        std::vector<Variable::t> HM ( xdata.rows() *ydata.rows() );
        std::vector<Variable::t> HP ( xdata.rows() *ydata.rows() );



        M->constraint ( Expr::sum ( gamma ),Domain::equalsTo ( 1.0 ) );


//      need to work with vectors of shared pointers of gurobi variables

// //      do this for each x coordinate
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                const unsigned int m ( LX[k].cols() * LY[l].cols() );
                const unsigned int currind ( k*ydata.rows() + l );

                HH[currind]   = M->variable ( new_array_ptr<int, 1> ( { ( int ) LX[k].cols(), ( int ) LY[l].cols() } ), Domain::inRange ( -100.0, 100.0 ) );
                HP[currind]   = M->variable ( new_array_ptr<int, 1> ( { ( int ) LX[k].cols(), ( int ) LY[l].cols() } ), Domain::greaterThan ( 0.0 ) );
                HM[currind]   = M->variable ( new_array_ptr<int, 1> ( { ( int ) LX[k].cols(), ( int ) LY[l].cols() } ), Domain::greaterThan ( 0.0 ) );

                Variable::t ht = Var::flatten ( HH[currind] );
                Variable::t hp = Var::flatten ( HP[currind] );
                Variable::t hm = Var::flatten ( HM[currind] );

                const auto prob_quadFormMatwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( prob_quadFormMat[currind].data(), shape ( m ) ) );
                const auto quadsqrt = std::make_shared<ndarray<double,1> > ( shape ( m ), [&] ( ptrdiff_t l ) {
                    return sqrt ( ( *prob_quadFormMatwrap ) ( l ) );
                } );
                const auto prob_vecwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( prob_vec[currind].data(), shape ( m ) ) );

                M->constraint ( Expr::vstack ( 0.5,vv->index ( currind ), Expr::mulElm ( quadsqrt,ht ) ), Domain::inRotatedQCone() );
                M->constraint ( Expr::vstack ( Expr::mul ( 0.5,uu->index ( currind ) ), gamma->index ( currind ), ht ), Domain::inRotatedQCone() );
                M->constraint ( Expr::sub ( summer->index ( currind ),Expr::mul ( 2.0,Expr::dot ( prob_vecwrap, ht ) ) ),Domain::equalsTo ( 0.0 ) );

                M->constraint ( Expr::sub ( ht,Expr::sub ( hp,hm ) ),Domain::equalsTo ( 0.0 ) );
//                 this is transposed
                const auto aawrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( aa[currind].data(), shape ( m ) ) );
                const auto bbwrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( bb[currind].data(), shape ( m ) ) );

                M->constraint ( Expr::sum ( Expr::sub ( Expr::mulElm ( hp,aawrap ),Expr::mulElm ( hm,bbwrap ) ) ),Domain::lessThan ( p ) ); // positivity


                eigenVector LXmeans ( LX[k].colwise().mean() );

                const auto LXmeanswrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( LXmeans.data(), shape ( LXmeans.size() ) ) );

                eigenVector LYmeans ( LY[l].colwise().mean() );
                const auto LYmeanswrap = std::shared_ptr<ndarray<double, 1> > ( new ndarray<double, 1> ( LYmeans.data(), shape ( LYmeans.size() ) ) );

                M->constraint ( Expr::mul ( LXmeanswrap,HH[currind] ), Domain::equalsTo ( 0.0 ) );
                M->constraint ( Expr::mul ( HH[currind],LYmeanswrap ), Domain::equalsTo ( 0.0 ) );
            }
        }


        M->objective ( ObjectiveSense::Minimize, Expr::sum ( Expr::add ( Expr::add ( vv,summer ), Expr::mul ( n*lam, uu ) ) ) );


        M->solve();
        if ( M->getPrimalSolutionStatus() == SolutionStatus::Optimal ) {
            const auto gammasol = * ( gamma->level() );
            for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                for ( unsigned int l = 0; l < ydata.rows(); ++l ) {
                    const unsigned int currind ( k*ydata.rows() + l );
                    gam ( currind ) = gammasol[currind];
                    const size_t m ( LX[k].cols() * LY[l].cols() );
                    const auto htsol = * ( HH[currind]->level() );
                    eigenVector aux ( m );
                    for ( size_t i = 0; i < m; ++i ) {
                        aux ( i ) = htsol[i];
                    }
//                     h[currind]= Q[currind]*aux;
                    Eigen::Map<eigenMatrix> auxmat ( aux.data(), LY[l].cols() ,LX[k].cols() );
                    H[currind] = Qy[l] * auxmat * Qx[k].transpose();
                    h[currind] = Eigen::Map<eigenVector>(H[currind].data(), H[currind].rows()*H[currind].cols());
                }
            }
//             std::cout << h.transpose() << std::endl;
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }
        return ( EXIT_SUCCESS );
    }
};
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
constexpr double DistributionEmbeddingTensorMKL<KernelMatrix, LowRank, KernelBasis>::p;

template<typename KernelMatrix, typename LowRank, typename KernelBasis>
constexpr double DistributionEmbeddingTensorMKL<KernelMatrix, LowRank, KernelBasis>::ALPHATRESH;
} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif // DISTRIBUTIONEMBEDDINGTENSORMKL_H_INCLUDED
