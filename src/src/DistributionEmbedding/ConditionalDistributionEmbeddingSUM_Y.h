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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_CONDITIONALDISTRIBUTIONEMBEDDINGSUM_Y_H_
#define RRCA_DISTRIBUTIONEMBEDDING_CONDITIONALDISTRIBUTIONEMBEDDINGSUM_Y_H_



namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{

//     this is the traditionalconditional distribution embedding. every coordinate in X has its own kernel. Y has a sum kernel
template<typename KernelMatrix, typename LowRank, typename KernelBasis,typename KernelMatrixY = KernelMatrix, typename LowRankY = LowRank, typename KernelBasisY = KernelBasis>
class ConditionalDistributionEmbeddingSUM_Y
{
    typedef typename KernelMatrix::value_type value_type;



public:
    ConditionalDistributionEmbeddingSUM_Y ( const Matrix& xdata_, const Matrix& ydata_ ) :
        xdata ( xdata_ ),
        xdata_t ( xdata_.rows() ),
        ydata ( ydata_ ),
        h ( xdata_.rows() ),
        H ( xdata_.rows() ),
        gam ( xdata_.rows() ),
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
        basy ( Ky,pivy ),
        lowerX ( xdata_.rows() ),
        upperX ( xdata_.rows() ),
        midX ( xdata_.rows() )
    {
        for ( unsigned int i = 0; i < xdata_.rows(); ++i ) {
            xdata_t[i] = xdata_.row ( i );
            Kx[i]   =  std::make_shared<KernelMatrix> ( xdata_t[i] );
            basx[i] = std::make_shared<KernelBasis> ( * ( Kx[i] ), pivx[i] );
        }
    }
    const std::vector<Matrix>& getH() const
    {
        return ( H );
    }

    /*
    *    \brief solves the full problem unconstrained where the x kernel is the direct sum of the one function and H_X
    */
    template<typename l1type>
    int solveFullUnconstrained ( l1type l1, double lam )
    {
        Matrix pain;
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            Kx[k]->kernel().l = l1;
            basx[k]->initFull ( * ( Kx[k] ), pivx[k].pivots() );
            pain = basx[k]->matrixKpp();
            pain.diagonal().array() += xdata.cols() * lam;
            H[k] = pain.template selfadjointView<Eigen::Lower>().llt().solve ( Eigen::MatrixXd::Identity ( xdata.cols(), xdata.cols() ) );
            h[k] = H[k].reshaped();
        }



        return ( EXIT_SUCCESS );
    }





    /*
    *    \brief solves the low-rank problem with structural consstraints
    */
    template<typename l1type, typename l2type>
    int solve ( l1type l1, l2type l2,double prec, double lam )
    {
        precomputeKernelMatrices ( l1,  l2, prec,  lam );
#ifdef RRCA_HAVE_MOSEK
        return ( solveMosek() );
#endif
        return ( EXIT_FAILURE );
    }





    /*
    *    \brief solves the low-rank problem unconstrained
    */
    int solveUnconstrained ( double l1, double l2,double prec,double lam )
    {

        precomputeKernelMatrices ( l1, l2,prec,lam );
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            h[k] = ( Qy * ( prob_vec[k].cwiseQuotient ( prob_quadFormMat[k] ).reshaped ( Qy.cols(), Qx[k].cols() ) ) * Qx[k].transpose() ).reshaped();
            H[k] = h[k].reshaped ( Qy.cols(), Qx[k].cols() );
        }

        return ( EXIT_SUCCESS );
    }


    /*
    *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
    */
    Matrix condExpfVec ( const Matrix& Xs ) const
    {
        Matrix vec = Matrix::Zero ( LY.cols(),Xs.cols() );
        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
//             const Matrix Kxmultsmall  = basx[k]->eval ( Xs.row(k) ).transpose();
            vec += H[k] * basx[k]->eval ( Xs.row(k) ).transpose();
        }


        return(vec/static_cast<double>(xdata.rows()));
    }

    /*
    *    \brief computes conditional expectation in one dimension in Y
    */
    Matrix condExpfY_X ( const std::vector<std::function<double ( const Vector& ) > >& funs, const Matrix& Xs ) const
    {
        const int funsize ( funs.size() );
        Matrix fy = ydata ( Kyblock[0].rows(),funsize );
        for ( unsigned int k = 0; k < funsize; ++k ) {
            for ( unsigned int l = 0; l < ydata.cols(); ++l ) {
                fy ( l,k ) = funs[k] ( ydata.col ( l ) );
            }
        }
//      now compute a matrix with all the function values

        return ( fy.transpose() ( Eigen::all,pivy.pivots() ) * condExpfVec ( Xs ) );


    }

    /*
    *    \brief computes conditional expectation in one dimension in Y
    */
    Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs ) const
    {
//      now compute a matrix with all the function values

        return ( Ys ( Eigen::all,pivy.pivots() ) * condExpfVec ( Xs ) );


    }

    const iVector& getYPivots() const
    {
        return ( pivy.pivots() );
    }


private:
    const Matrix& xdata;
    std::vector<Matrix> xdata_t;
    const Matrix& ydata;

    std::vector<Vector> h; // the vector of coefficients
    std::vector<Matrix> H; // the vector of coefficient matrices
    Vector gam; // this is alpha times beta

    std::vector<Matrix> LX; // the important ones
    Matrix LY; // the important ones

    std::vector<Vector> Xvar; // the sample matrix of the important X ones
    Vector Yvar; // the sample matrix of the important X ones

    std::vector<Matrix> Kxblock; // the kernelfunctions of the important X ones
    Matrix Kyblock; // the kernelfunctions of the important X ones

    std::vector<Matrix> Qx; // the basis transformation matrix
    Matrix Qy; // the basis transformation matrix
//     std::vector<Matrix> Q;

    std::vector<Vector> prob_quadFormMat; // this is a vector, because we only need the diagonal
    std::vector<Vector> prob_vec; // the important ones

    std::vector<std::shared_ptr<KernelMatrix> > Kx;
    KernelMatrixY Ky;

    std::vector<LowRank>  pivx;
    LowRankY  pivy;

    std::vector<std::shared_ptr<KernelBasis> > basx;
    KernelBasisY basy;

    std::vector<Vector> lowerX; //lower bound
    std::vector<Vector> upperX; //upper bound
    std::vector<Vector> midX; //mean

    Vector lowerY; //lower bound
    Vector upperY; //upper bound
    Vector midY; //mean

    double tol;

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices ( double l1, double l2,double prec, double lam )
    {
        tol = prec;
        Ky.kernel().l = l2;
        pivy.compute ( Ky, tol );
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

        precomputeHelper ( lam );
//         std::cout << "mx " << Kxblock.cols() << " my " << Kyblock.cols() << std::endl;
    }





    void precomputeHelper ( double lam )
    {

        lowerY = LY.colwise().minCoeff();
        midY = LY.colwise().mean();
        upperY = LY.colwise().maxCoeff();

        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
            lowerX[k] = LX[k].colwise().minCoeff();
            midX[k] = LX[k].colwise().mean();
            upperX[k] = LX[k].colwise().maxCoeff();
            prob_vec[k] = ( LY.transpose() * LX[k] ).reshaped();

            Xvar[k] = ( LX[k].transpose() * LX[k] ).diagonal();
            prob_quadFormMat[k]= Xvar[k].transpose().replicate ( LY.cols(),1 ).reshaped().array()+LY.rows() *lam;
        }

    }
#ifdef RRCA_HAVE_MOSEK
//     int solveFullUnconstrainedMosek(double l1, double l2, double lam){
//         using namespace mosek::fusion;
//         using namespace monty;
//
//         Model::t M = new Model ( "ConditionalDistributionEmbeddingFull" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
//         auto _M = finally ( [&]() {
//             M->dispose();
//         } );
//
//         Kx.kernel().l = l1;
//         basx.initFull(Kx);
//
//         Ky.kernel().l = l2;
//         basy.initFull(Ky);
//
//         Kxblock = basx.matrixKpp();
//         Kyblock = basy.matrixKpp();
//
//         Matrix oida = Kxblock;
//         oida.diagonal().array() += lam *Kxblock.rows();
//
//         Eigen::SelfAdjointEigenSolver<Matrix> eigx(Kxblock*oida);
//
//         LX = eigx.eigenvectors() * eigx.eigenvalues().cwiseSqrt().asDiagonal() * eigx.eigenvectors().transpose();
//
//         Eigen::SelfAdjointEigenSolver<Matrix> eigy(Kyblock);
//         LY = eigy.eigenvectors() * eigy.eigenvalues().cwiseSqrt().asDiagonal() * eigy.eigenvectors().transpose();
//
//         Matrix L(Kxblock.cols()*Kyblock.cols(),Kxblock.cols()*Kyblock.cols());
//         for(unsigned int i = 0; i < LY.cols(); ++i){
//             for(unsigned int j = 0; j < LY.cols(); ++j){
//                 L.block(i*LY.cols(),j*LY.cols(),LY.cols(),LY.cols()) = LX(i,j)*LY;
//             }
//         }
//
// //         can define matrix variable for the bilinear form ,need to keep it row major here, therefore we define H transposed
//         M_Variable::t HH_t = M->variable("H", new_array_ptr<int, 1>({(int)Kxblock.cols(), (int)Kyblock.cols()}), M_Domain::unbounded());
//         M_Variable::t ht = Var::flatten( HH_t ); // this will be correct because of the row major form of HH_t
//
// //         for the quadratic cone
//         M_Variable::t uu = M->variable( "uu", M_Domain::greaterThan(0.0));
//
//         Matrix proddo =   Kyblock*Kxblock;
//
//
//          const Matrix::t Lwrap = Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2( L.data(), shape ( L.rows(), L.cols() ) ) ) );
//
//          auto prod_wrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( proddo.data(), shape ( proddo.rows()* proddo.cols() ) ) ) ;
//
//         M->constraint(M_Expr::vstack(0.5, uu, M_Expr::mul(Lwrap,ht)), M_Domain::inRotatedQCone()); // quadratic cone for objective function
//
//         M->objective ( ObjectiveSense::Minimize, M_Expr::add(uu,M_Expr::mul ( -2.0,M_Expr::dot(prod_wrap,ht))));
//         M->solve();
//         if ( M->getPrimalSolutionStatus() == SolutionStatus::Optimal) {
//             const unsigned int m = Kyblock.rows()*Kyblock.rows();
//             Vector aux ( m );
//             const auto htsol = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( * ( ht->level() ) ) );
//             for ( size_t i = 0; i < m; ++i ) {
//                 aux ( i ) = (*htsol)(i);
//             }
//             Eigen::Map<Matrix> auxmat ( aux.data(), LY.cols() ,LX.cols() );
//             H = auxmat;
//             h = H.reshaped();
//         } else {
//             std::cout << "infeasible  " <<  std::endl;
//             return ( EXIT_FAILURE );
//         }
//         return ( EXIT_SUCCESS );
//     }

    int solveMosek()
    {
        M_Model M = new mosek::fusion::Model ( "ConditionalDistributionEmbeddingSUM_Y" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
        
        //         for the quadratic cone
            M_Variable::t uu = M->variable ( "uu", xdata.rows(),M_Domain::greaterThan ( 0.0 ) );
            M_Variable::t summer = M->variable ( "summer", xdata.rows() , M_Domain::unbounded() );
            std::vector<M_Variable::t> HH_t ( xdata.rows()  );
            Vector Qycolsum = Qy.colwise().sum();
            const auto Qycolsum_wrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( Qycolsum.data(), shape ( Qy.cols() ) ) );
            const M_Matrix::t Qy_twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2( Qy.data(), monty::shape ( Qy.cols(), Qy.rows() ) ) ) );


        for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
//         can define matrix variable for the bilinear form ,need to keep it row major here, therefore we define H transposed
            HH_t[k] = M->variable ( monty::new_array_ptr<int, 1> ( { ( int ) LX[k].cols(), ( int ) LY.cols() } ), M_Domain::unbounded() );
            M_Variable::t ht = M_Var::flatten ( HH_t[k] ); // this will be correct because of the row major form of HH_t


            


            const auto prob_quadFormMatwrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( prob_quadFormMat[k].data(), monty::shape ( LX[k].cols()*LY.cols() ) ) );
            const auto quadsqrt = std::make_shared<ndarray<double,1> > ( shape ( LX[k].cols()*LY.cols() ), [&] ( ptrdiff_t l ) {
                return sqrt ( ( *prob_quadFormMatwrap ) ( l ) );
            } );
            Matrix Lxmean = basx[k]->matrixKpp() * Qx[k];
            const M_Matrix::t Lx_twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2( Lxmean.data(), monty::shape ( Lxmean.cols(), Lxmean.rows() ) ) ) );
            

            M->constraint ( M_Expr::vstack ( 0.5, uu->index(k), M_Expr::mulElm ( quadsqrt,ht ) ), M_Domain::inRotatedQCone() ); // quadratic cone for objective function

//             M->constraint ( M_Expr::mul ( M_Expr::mul ( Lx_twrap->transpose(), HH_t ),Qycolsum_wrap ), M_Domain::equalsTo ( 1.0 ) );


            const auto XminWrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( lowerX[k].data(), shape ( lowerX[k].size() ) ) );
            const auto XmeanWrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( midX[k].data(), shape ( midX[k].size() ) ) );
            const auto XmaxWrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( upperX[k].data(), shape ( upperX[k].size() ) ) );

//             M->constraint ( M_Expr::mul ( M_Expr::mul ( XminWrap, HH_t ),Qy_twrap ), M_Domain::greaterThan ( 0.0 ) );
//             M->constraint ( M_Expr::mul ( M_Expr::mul ( XmeanWrap, HH_t ),Qy_twrap ), M_Domain::greaterThan ( 0.0 ) );
//             M->constraint ( M_Expr::mul ( M_Expr::mul ( XmaxWrap, HH_t ),Qy_twrap ), M_Domain::greaterThan ( 0.0 ) );


//         this is p almost sure constraint
//         M->constraint(M_Expr::mul(M_Expr::mul(Lx_twrap->transpose(), HH_t),Qy_twrap), M_Domain::greaterThan(0.0));


            const auto prob_vecwrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( prob_vec[k].data(), monty::shape ( LX[k].cols()*LY.cols() ) ) );
            M->constraint ( M_Expr::sub ( summer->index ( k ),M_Expr::mul (  - 2.0,M_Expr::dot ( prob_vecwrap, ht ) ) ),M_Domain::equalsTo ( 0.0 ) );
        }
        //         !!!!!!!!!

        M->objective ( ObjectiveSense::Minimize, M_Expr::sum(M_Expr::add ( uu,summer ) ));
        M->solve();
        if ( M->getPrimalSolutionStatus() == SolutionStatus::Optimal ) {
            for ( unsigned int k = 0; k < xdata.rows(); ++k ) {
                    const unsigned int currind ( k);
                    gam ( currind ) = 1.0;
                    const auto htsol = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( * ( HH_t[currind]->level() ) ) );
                    const Eigen::Map<Matrix> auxmat ( htsol->raw(), LY.cols() ,LX[k].cols() );
                    H[currind] = Qy * auxmat * Qx[k].transpose();
                    h[currind] = H[currind].reshaped();
            }
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
