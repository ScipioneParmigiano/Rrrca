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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_CONDITIONALDISTRIBUTIONEMBEDDING_H_
#define RRCA_DISTRIBUTIONEMBEDDING_CONDITIONALDISTRIBUTIONEMBEDDING_H_


// #include<eigen3/Eigen/Dense>
namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{
    
//     this is the traditionalconditional distribution embedding 
template<typename KernelMatrix, typename LowRank, typename KernelBasis,typename KernelMatrixY = KernelMatrix, typename LowRankY = LowRank, typename KernelBasisY = KernelBasis>
class ConditionalDistributionEmbedding
{


public:
    ConditionalDistributionEmbedding ( const Matrix& xdata_, const Matrix& ydata_) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        Kx ( xdata_ ),
        Ky ( ydata_ ),
        basx(Kx,pivx),
        basy(Ky,pivy)
    {


    }
    const Matrix& getH() const {
        return(H);
    }
    
    unsigned int getSubspaceDimension() const {
        return(LX.cols() + LY.cols());
    }
    
      /*
   *    \brief solves the full problem unconstrained where the x kernel is the direct sum of the one function and H_X
   */
    template<typename l1type>
    int solveFullUnconstrained(l1type l1, double lam){
        Kx.kernel().l = l1;
        basx.initFullLower(Kx);
        
        Matrix pain = basx.matrixKpp();
        pain.diagonal().array() += xdata.cols()* lam;
       

        H = pain.template selfadjointView<Eigen::Lower>().llt().solve ( Matrix::Identity ( xdata.cols(), xdata.cols() ) );
        h = H.reshaped();
        return(EXIT_SUCCESS);
    }
    

  
  
    
      /*
   *    \brief solves the low-rank problem with structural consstraints
   */
    template<typename l1type, typename l2type>
    int solve(l1type l1, l2type l2,double prec, double lam){
        precomputeKernelMatrices( l1,  l2, prec,  lam);
#ifdef RRCA_HAVE_MOSEK
                return(solveMosek());
#endif
        return(EXIT_FAILURE);
    }
    

    
    
    
      /*
   *    \brief solves the low-rank problem unconstrained 
   */
        int solveUnconstrained ( double l1, double l2,double prec,double lam )
    {
        
        precomputeKernelMatrices ( l1, l2,prec,lam );
        h = (Qy * ( prob_vec.cwiseQuotient (prob_quadFormMat).reshaped(Qy.cols(), Qx.cols()) ) * Qx.transpose()).reshaped();
        H = h.reshaped(Qy.cols(), Qx.cols());

        return ( EXIT_SUCCESS );
    }
 
    
        /*
   *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
   */
    Matrix condExpfVec ( const Matrix& Xs ) const
    {
        const Matrix& Kxmultsmall  = basx.eval(Xs).transpose();

        return H * Kxmultsmall;
    }
    
        /*
    *    \brief computes conditional expectation in one dimension in Y
    */
    Matrix condExpfY_X ( const std::vector<std::function<double ( const Vector& ) > >& funs, const Matrix& Xs ) const {
        const int funsize ( funs.size() );
        Matrix fy = ydata ( Kyblock.rows(),funsize );
        for ( unsigned int k = 0; k < funsize; ++k ) {
            for ( unsigned int l = 0; l < ydata.cols(); ++l ) {
                fy ( l,k ) = funs[k] ( ydata.col ( l ) );
            }
        }
//      now compute a matrix with all the function values

        return ( fy.transpose()(Eigen::all,pivy.pivots()) * condExpfVec(Xs) );
        
        
    }
    
            /*
    *    \brief computes conditional expectation in one dimension in Y
    */
    Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs ) const {
//      now compute a matrix with all the function values

        return ( Ys(Eigen::all,pivy.pivots()) * condExpfVec(Xs) );
        
        
    }
    
    const iVector& getYPivots() const {
        return(pivy.pivots());
    }
    
    bool positiveOnGrid() const {
        std::cout << H * Kxblock(pivx.pivots(),Eigen::all).transpose() << std::endl;
        return(static_cast<bool>((H * Kxblock.transpose()).minCoeff()>= 0.0));
    }
    

private:
    const Matrix& xdata;
    const Matrix& ydata;
    
    Vector h; // the vector of coefficients
    Matrix H; // the matrix of coefficients, h=vec H

    Matrix LX; // the important ones
    Matrix LY; // the important ones

    Vector Xvar; // the sample matrix of the important X ones
    Vector Yvar; // the sample matrix of the important X ones

    Matrix Kxblock; // the kernelfunctions of the important X ones
    Matrix Kyblock; // the kernelfunctions of the important X ones
//     Matrix Kyblock_m; // the kernelfunctions of the important X ones accrdog to tensor prod rule
//     Matrix Kxblock_m; // the kernelfunctions of the important X ones accrdog to tensor prod rule

    Matrix C_Under; // the important ones
    Matrix C_Over; // the important ones

    Matrix Qx; // the basis transformation matrix
    Matrix Qy; // the basis transformation matrix
//     Matrix Q;
    
    Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
    Vector prob_vec; // the important ones
    
    
    Vector xbound_min;
    Vector xbound_max;
    
    Vector ybound_min;
    Vector ybound_max;
    
    Matrix Qy_pos; // the important ones
    Matrix Qy_neg; // the important ones

    KernelMatrix Kx;
    KernelMatrixY Ky;

    LowRank pivx;
    LowRankY pivy;
    
    KernelBasis basx;
    KernelBasisY basy;
    

    /*
   *    \brief computes the kernel basis and tensorizes it
   */ 
    void precomputeKernelMatrices ( double l1, double l2,double prec, double lam )
    {
                Kx.kernel().l = l1;
        Ky.kernel().l = l2;

        // pivx.compute ( Kx,prec,0,RRCA_LOWRANK_STEPLIM  );
        // pivy.compute ( Ky,prec,0,RRCA_LOWRANK_STEPLIM  );   
        
        pivx.compute ( Kx,prec);
        pivy.compute ( Ky,prec); 
        
        // pivy.computeBiorthogonalBasis();
        
        basx.init(Kx, pivx.pivots());
        basx.initSpectralBasisWeights(pivx);

        
        
        Qx =  basx.matrixQ() ;
        Kxblock = basx.eval(xdata);
        
        

        LX = Kxblock * Qx;
        
        basy.init(Ky, pivy.pivots());
        basy.initSpectralBasisWeights(pivy);
        
        

        
        Qy =  basy.matrixQ() ;
        Kyblock = basy.eval(ydata);
        LY = Kyblock * Qy;
        
//         Qy_pos = Qy.cwiseMax(0.0);
//         Qy_neg = Qy.cwiseMin(0.0).cwiseAbs();
//         
//         xbound_min = LX.colwise().minCoeff().transpose();
//         xbound_max = LX.colwise().maxCoeff().transpose();
        
        precomputeHelper(lam);
    }
    

    
    
    
    void precomputeHelper(double lam){
        const unsigned int n ( LX.rows() );
        const unsigned int  rankx ( LX.cols() );
        const unsigned int  ranky ( LY.cols());
        const unsigned int  m ( rankx*ranky );
        
       
        prob_vec = (LY.transpose() * LX).reshaped();

        Xvar =  ( LX.transpose() * LX).diagonal();

        Matrix oida = Xvar.transpose().replicate(ranky,1);
        prob_quadFormMat= oida.reshaped().array()+n*lam;

    }
#ifdef RRCA_HAVE_MOSEK

    
    int solveMosek(){
        M_Model M = new mosek::fusion::Model ( "ConditionalDistributionEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
        unsigned int m = LX.cols()*LY.cols();
       
//         can define matrix variable for the bilinear form ,need to keep it row major here, therefore we define H transposed
        M_Variable::t HH_t = M->variable("H", monty::new_array_ptr<int, 1>({(int)LX.cols(), (int)LY.cols()}), M_Domain::unbounded());
        M_Variable::t ht = M_Var::flatten( HH_t ); // this will be correct because of the row major form of HH_t

//         for the quadratic cone
        M_Variable::t uu = M->variable( "uu", M_Domain::greaterThan(0.0));

        Vector Qycolsum = Qy.colwise().sum();
        auto Qycolsum_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( Qycolsum.data(), monty::shape ( Qy.cols()) ) );
         
        
        auto quadsqrt = std::make_shared<M_ndarray_1>( monty::shape ( m ), [&] ( ptrdiff_t l ) { return sqrt ( prob_quadFormMat  ( l )); } );
         Vector Lxmean = LX.colwise().mean().transpose();
         
          auto Lxmean_twrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( Lxmean.data(), monty::shape ( Lxmean.size()) ) );
         
         // const M_Matrix::t Lxmean_twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2 ( Lxmean.data(), monty::shape ( Lxmean.cols(), Lxmean.rows() ) ) ) );
         // const M_Matrix::t Ly_twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2 ( LY.data(), monty::shape ( LY.cols(), LY.rows() ) ) ) );
         const M_Matrix::t Qy_twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2  ( Qy.data(), monty::shape ( Qy.cols(), Qy.rows() ) ) ) );

        M->constraint(M_Expr::vstack(0.5, uu, M_Expr::mulElm(quadsqrt,ht)), M_Domain::inRotatedQCone()); // quadratic cone for objective function
        
        M->constraint(M_Expr::dot(M_Expr::mul(Lxmean_twrap, HH_t),Qycolsum_wrap), M_Domain::equalsTo(1.0)); 
        
        Vector Lxpiv = LX.colwise().sum();
        
        auto Lxpiv_wrap_t = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( Lxpiv.data(), monty::shape ( Lxpiv.size()) ) );
        M->constraint(M_Expr::mul(M_Expr::mul(Lxpiv_wrap_t,HH_t),Qy_twrap), M_Domain::greaterThan(0.0));

            // Matrix LXsub = LX.topRows(std::min(static_cast<Eigen::Index>(RRCA_LOWRANK_STEPLIM),xdata.cols()));
            // const M_Matrix::t Lx_twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2  ( LXsub.data(), monty::shape ( LXsub.cols(), LXsub.rows() ) ) ) );
            // M->constraint(M_Expr::mul(M_Expr::mul(Lx_twrap->transpose(), HH_t),Qy_twrap), M_Domain::greaterThan(0.0));
        
        // M_Variable::t HH_m = M->variable("Hm", monty::new_array_ptr<int, 1>({(int)LX.cols(), (int)LY.cols()}), M_Domain::greaterThan(0.0));
        // M_Variable::t HH_p = M->variable("Hp", monty::new_array_ptr<int, 1>({(int)LX.cols(), (int)LY.cols()}), M_Domain::greaterThan(0.0));
        
        // M->constraint(M_Expr::sub(M_Expr::sub(HH_p,HH_m), HH_t), M_Domain::equalsTo(0.0));
            
            // const M_Matrix::t Qy_poswrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( Qy_pos.data(), monty::shape ( Qy_pos.cols(), Qy_pos.rows()) ) ));
            // const M_Matrix::t Qy_negwrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( Qy_neg.data(), monty::shape ( Qy_neg.cols(), Qy_neg.rows()) ) ));
            
            // auto xmin_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( xbound_min.data(), monty::shape ( xbound_min.size()) ) );
            // auto xmax_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( xbound_max.data(), monty::shape ( xbound_max.size()) ) );
            
            // M->constraint(M_Expr::sub(M_Expr::add(M_Expr::mul(xmin_wrap,M_Expr::mul(HH_p,Qy_poswrap)),M_Expr::mul(xmin_wrap,M_Expr::mul(HH_m,Qy_negwrap))),
                                      // M_Expr::add(M_Expr::mul(xmax_wrap,M_Expr::mul(HH_p,Qy_negwrap)),M_Expr::mul(xmax_wrap,M_Expr::mul(HH_m,Qy_poswrap)))),M_Domain::greaterThan ( 0.0 ));
//         
            // const M_Matrix::t C_underwrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( C_Under.data(), monty::shape ( C_Under.cols(), C_Under.rows()) ) ));
            // const M_Matrix::t C_overwrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( C_Over.data(), monty::shape ( C_Over.cols(), C_Over.rows()) ) ));
            // M->constraint(M_Expr::sub(M_Expr::dot(C_underwrap,HH_p),M_Expr::dot(C_overwrap,HH_m)),M_Domain::greaterThan ( 0 ));

         auto prob_vecwrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1 ( prob_vec.data(), monty::shape ( m) ) ); 

        //         !!!!!!!!!
        
        M->objective (  mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(uu,M_Expr::mul ( -2.0,M_Expr::dot(prob_vecwrap,ht))));
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
            const unsigned int m = LX.cols()*LY.cols();
            Vector aux ( m );
            M_ndarray_1 htsol   = * ( ht->level() );
            
//             h= Q*aux;
            Eigen::Map<Matrix> auxmat ( htsol.raw(), LY.cols() ,LX.cols() );
            H = Qy * auxmat * Qx.transpose();
//             H = auxmat;
            h = H.reshaped();
//             std::cout << h.transpose() << std::endl;
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
