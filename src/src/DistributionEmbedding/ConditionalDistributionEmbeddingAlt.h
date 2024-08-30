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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_CONDITIONALDISTRIBUTIONEMBEDDINGALT_H_
#define RRCA_DISTRIBUTIONEMBEDDING_CONDITIONALDISTRIBUTIONEMBEDDINGALT_H_


// #include<eigen3/Eigen/Dense>
namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{
    
//     this is the traditionalconditional distribution embedding 
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class ConditionalDistributionEmbeddingAlt
{


public:
    ConditionalDistributionEmbeddingAlt ( const Matrix& xdata_, const Matrix& ydata_) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        Kx ( xdata_ ),
        basx(Kx,pivx)
    {

    }
    const Matrix& getH() const {
        return(H);
    }
    unsigned int getSubspaceDimension() const {
        return(LX.cols());
    }
    
      /*
   *    \brief solves the full problem unconstrained where the x kernel is the direct sum of the one function and H_X
   */
    template<typename l1type>
    int solveFullUnconstrained(l1type l1, double lam){
        Kx.kernel().l = l1;
        Matrix pain;
            basx.initFullLower(Kx);
            pain = basx.matrixKpp();
        

        
        pain.diagonal().array() += pain.cols()* lam;
       

        H = pain.template selfadjointView<Eigen::Lower>().llt().solve ( Matrix::Identity ( pain.cols(), pain.cols() ) );
        return(EXIT_SUCCESS);
    }
    
          /*
   *    \brief solves the low-rank problem unconstrained 
   */
        int solveUnconstrained ( double l1, double prec,double lam )
    {
        
        precomputeKernelMatrices ( l1, prec,lam );
//         const double n(LX.rows());
         
        
        H = idealH * Qx.transpose() ;

        return ( EXIT_SUCCESS );
    }
    

  
  
    
      /*
   *    \brief solves the low-rank problem with structural consstraints
   */
    template<typename l1type>
    int solve(l1type l1, double prec, double lam){
        precomputeKernelMatrices( l1,   prec,  lam);
#ifdef RRCA_HAVE_MOSEK
                return(solveMosek());
#endif
        return(EXIT_FAILURE);
    }
    
    
//         /*
//    *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
//    */
//     Matrix condExpfVecFull ( const Matrix& Xs ) const
//     {
//         const Matrix& Kxmultsmall  = basx.eval(Xs).transpose();
// 
//         return H * Kxmultsmall;
//     }
    
    Matrix condExpfVec ( const Matrix& Xs ) const
    {
        const Matrix& Kxmultsmall  = basx.eval(Xs).transpose();

        return H * Kxmultsmall;
    }
    
        /*
    *    \brief computes conditional expectation in one dimension in Y
    */

    
            /*
    *    \brief computes conditional expectation in one dimension in Y
    */
    Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs ) const {
//      now compute a matrix with all the function values

        return ( Ys * H * basx.eval(Xs).transpose() );
        
        
    }
    
      const iVector& getYPivots() const {
        return(pivots);
    }
    

    

private:
    const Matrix& xdata;
    const Matrix& ydata;


    Vector h; // the vector of coefficients
    Matrix H; // the matrix of coefficients, h=vec H
    Matrix idealH; 

    Matrix LX; // the important ones

    Vector Xvar; // the sample matrix of the important X ones

    Matrix Kxblock; // the kernelfunctions of the important X ones
//     Matrix Kyblock_m; // the kernelfunctions of the important X ones accrdog to tensor prod rule
//     Matrix Kxblock_m; // the kernelfunctions of the important X ones accrdog to tensor prod rule

    Matrix Qx; // the basis transformation matrix
//     Matrix Q;
    
    KernelMatrix Kx;

    LowRank pivx;
    
    KernelBasis basx;
    
    Vector prob_quadFormMat;
    
    
    iVector pivots;
// this one is used for subsampling the positivity constraint
    


    /*
   *    \brief computes the kernel basis and tensorizes it
   */ 
    void precomputeKernelMatrices ( double l1, double prec, double lam )
    {
        Kx.kernel().l = l1;

        // pivx.compute ( Kx,prec,0,RRCA_LOWRANK_STEPLIM  );
        
        // std::cout << "starting cholesky" << std::endl;
        pivx.compute ( Kx,prec);
        
        basx.init(Kx, pivx.pivots());
            basx.initSpectralBasisWeights(pivx);
            Kxblock = basx.eval(xdata);
        
        
        Qx =  basx.matrixQ() ;
        LX = Kxblock * Qx;
        // std::cout << "ending cholesky with dimension " << LX.cols() << std::endl;
        
        
        
        precomputeHelper(lam);
    }
    

    
    
    
    void precomputeHelper(double lam){
        const unsigned int n ( LX.rows() );


            Xvar =  ( LX.transpose() * LX).diagonal();

            prob_quadFormMat= Xvar.array()+n*lam;
            idealH = LX * prob_quadFormMat.cwiseInverse().asDiagonal();

    }
#ifdef RRCA_HAVE_MOSEK

    
    int solveMosek(){
        M_Model M = new mosek::fusion::Model ( "ConditionalDistributionEmbeddingAlt" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
       
//         can define matrix variable for the bilinear form ,need to keep it row major here, therefore we define H transposed
        M_Variable::t HH_t = M->variable("H", monty::new_array_ptr<int, 1>({(int)idealH.cols(), (int)idealH.rows()}), M_Domain::unbounded());
        M_Variable::t hh_t = M_Var::flatten(HH_t);
//         for the quadratic cone
        M_Variable::t uu = M->variable( "uu", M_Domain::greaterThan(0.0));

        Vector Qxcolsum = Qx.colwise().sum();
        auto Qxcolsum_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1  ( Qxcolsum.data(), monty::shape ( Qxcolsum.size()) ) );
        Vector LXmean = LX.colwise().mean();
        
        const M_Matrix::t Qx_twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2  ( Qx.data(), monty::shape ( Qx.cols(), Qx.rows() ) ) ) );
         
        
//          Matrix Lxmean = basx.matrixKpp() * Qx;
        
         auto  Lx_meanwrap = std::shared_ptr<M_ndarray_1> (new  M_ndarray_1( LXmean.data(), monty::shape ( LXmean.size()) ) ) ;
//          Matrix idealHt = idealH.transpose();
         const M_Matrix::t idealWrap_t = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2 ( idealH.data(), monty::shape ( idealH.cols(), idealH.rows() ) ) ) );

        M->constraint(M_Expr::vstack(0.5, uu, M_Expr::flatten(M_Expr::sub(idealWrap_t,HH_t))), M_Domain::inRotatedQCone()); // quadratic cone for objective function
//         normalization constraint
        M->constraint(M_Expr::sum(M_Expr::mul(Lx_meanwrap,HH_t)), M_Domain::equalsTo(1.0)); 
//         positivity constraint
        Matrix LXsub = LX.topRows(std::min(static_cast<Eigen::Index>(RRCA_LOWRANK_STEPLIM),xdata.cols()));
        const M_Matrix::t Lx_twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2  ( LXsub.data(), monty::shape ( LXsub.cols(), LXsub.rows() ) ) ) );
            M->constraint(M_Expr::mul(Lx_twrap->transpose(),HH_t), M_Domain::greaterThan(0.0));
        
        
        
//         this is p almost sure constraint
//         M->constraint(M_Expr::mul(M_Expr::mul(Lx_twrap->transpose(), HH_t),Qy_twrap), M_Domain::greaterThan(0.0)); 
        

//         const auto prob_vecwrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1 ( prob_vec.data(), monty::shape ( m) ) ); 

        //         !!!!!!!!!
        
        M->objective (  mosek::fusion::ObjectiveSense::Minimize, uu);
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
             M_ndarray_1 htsol   = * ( hh_t->level() );
//             const auto htsol = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( * ( hh_t->level() ) ) );
            const Eigen::Map<Matrix> auxmat ( htsol.raw(), idealH.rows() ,idealH.cols() );
            H = auxmat * Qx.transpose();
//             H = auxmat;
//             h = H.reshaped();
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
