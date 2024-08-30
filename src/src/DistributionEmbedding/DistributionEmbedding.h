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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDING_H_
#define RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDING_H_

#include "../util/Stopwatch.h"
#include <iostream>

namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{
    
//     plain vanilla implementation, to be used only with SumKernel
//     !!! to be used only with SumKernel !!!
template<typename KernelMatrix, typename LowRank, typename KernelBasis,typename KernelMatrixY = KernelMatrix, typename LowRankY = LowRank, typename KernelBasisY = KernelBasis>
class DistributionEmbedding
{
    typedef typename KernelMatrix::value_type value_type;



public:
    DistributionEmbedding ( const Matrix& xdata_, const Matrix& ydata_) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        Kx ( xdata_ ),
        Ky ( ydata_ ),
        basx(Kx,pivx),
        basy(Ky,pivy)
    {

        std::cout << "constr: \n" << Kx.full() << std::endl;

    }
    const Matrix& getH() const {
        return(H);
    }
    
    unsigned int getSubspaceDimension() const {
        return(LX.cols() + LY.cols());
    }
  
    
      /*
   *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
   */
    template<typename l1type, typename l2type>
    int solve(l1type l1, l2type l2,double prec, double lam, RRCA::Solver solver = RRCA::Solver::Mosek, bool pointWisePositivity = false){
        precomputeKernelMatrices( l1,  l2, prec,  lam);
        switch(solver){
// #ifdef RRCA_HAVE_GUROBI
//             case Gurobi:
//                 return(solveGurobi());
// #endif
#ifdef RRCA_HAVE_MOSEK
            case Mosek:
                return(solveMosek(pointWisePositivity));
#endif
            default:
                return(EXIT_FAILURE);
        }
        return(EXIT_FAILURE);
    }
    
#ifdef RRCA_HAVE_MOSEK
//       /*
//    *    \brief this maximizes the log likelihood subject to regularization
//    */
//     int solveML(double l1, double l2,double prec, double lam){
//         precomputeKernelMatrices( l1,  l2, prec,  lam);
// 
//         M_Model M = new mosek::fusion::Model ( "DistributionEmbeddingML" );
// //         M->setLogHandler([=](const std::string & msg) {
// //             std::cout << msg << std::flush;
// //         } );
//         auto _M = monty::finally ( [&]() {
//             M->dispose();
//         } );
//         const int mx(LX.cols());
//         const int my(LY.cols());
//         const int n(LX.rows());
//         const int m(mx*my);
// 
// //      the variables
//         M_Variable::t HH = M->variable("H", monty::new_array_ptr<int,1>({mx,my}), M_Domain::unbounded() );// the matrix
//         M_Variable::t tt = M->variable("tt", n, M_Domain::unbounded() );// for the exponential cones
//         M_Variable::t x1 = M->variable("x1", n, M_Domain::greaterThan(0.0) );// for the exponential cones
//         M_Variable::t uu = M->variable( "uu", M_Domain::greaterThan(0.0));
// 
// //      compute the expectations of the bases, for the constraints that ensure that it is normalized
//         Vector yexp = LY.colwise().mean();
//         Vector xexp = LX.colwise().mean();
//         const auto yexp_w = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( yexp.data(), monty::shape ( my ) ) );
//         const auto xexp_w = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( xexp.data(), monty::shape ( mx ) ) );
// 
// //      the normalization
//         M->constraint(M_Expr::mul(xexp_w,HH),M_Domain::equalsTo(0.0));
//         M->constraint(M_Expr::mul(HH,yexp_w),M_Domain::equalsTo(0.0));
//         
//         
// 
// 
// //      the setting the tt variables
// //      now be careful for the wrappers, since Eigen is column major
// //      so the below are LY and LX transposed, respectively
//         const auto LY_w_t = std::shared_ptr<M_ndarray_2> ( new M_ndarray_2( LY.data(), monty::shape ( my,n ) ) );
//         const auto LX_w_t = std::shared_ptr<M_ndarray_2> ( new M_ndarray_2( LX.data(), monty::shape ( mx,n ) ) );
//         M_Matrix::t LY_t = M_Matrix::dense(LY_w_t);
//         M_Matrix::t LX_t = M_Matrix::dense(LX_w_t);
// 
// //      set x1
//         M->constraint(M_Expr::sub(x1,M_Expr::add(M_Expr::constTerm(n, p),M_Expr::mulDiag(M_Expr::mul(LX_t->transpose(), HH), LY_t))),M_Domain::equalsTo(0.0));
//                 
//         M_Variable::t hh = M_Var::flatten( HH );       
// 
//         M->constraint(M_Expr::vstack(0.5, uu, hh), M_Domain::inRotatedQCone()); // quadratic cone for objective function
//         
// //      now we check how many data points we have. For n<=200 let's do the actual problem with exponential cones
//         if(n<=10000){
// //             this puts all the exponential cones in one shot
//             M->constraint(M_Expr::hstack(x1, M_Expr::constTerm(n, 1.0), tt), M_Domain::inPExpCone());
//             M->objective ( mosek::fusion::ObjectiveSense::Maximize, M_Expr::sub(M_Expr::mul(M_Expr::sum ( tt),1.0/static_cast<double>(n)) , M_Expr::mul(n*lam, uu ))) ;
//         } else {
// //             M->constraint(tt, M_Domain::);
//             //             this is the first-order approximation of the exponential cone problem
//             M->objective ( mosek::fusion::ObjectiveSense::Maximize, M_Expr::sub(M_Expr::mul(M_Expr::sum ( x1),1.0/static_cast<double>(n)) , M_Expr::mul(n*lam, uu ))) ;
//         }
// //      now set in one go all the exponential cones
// 
// 
//         
//         M->solve();
//         auto ss = M->getPrimalSolutionStatus();
//         auto sd = M->getDualSolutionStatus();
// //         std::cout << "Solution status: " << ss << std::endl;
//         if(ss != mosek::fusion::SolutionStatus::Optimal || sd != mosek::fusion::SolutionStatus::Optimal) {
//             return(EXIT_FAILURE);
//         }
// 
//         Vector aux ( m );
//             const auto htsol = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( * ( HH->level() ) ) );
//             for ( size_t i = 0; i < m; ++i ) {
//                 aux ( i ) = (*htsol)(i);
//             }
// //             h= Q*aux;
//             Eigen::Map<Matrix> auxmat ( aux.data(), LY.cols() ,LX.cols() );
//             H = Qy * auxmat * Qx.transpose();
//             h = Eigen::Map<Vector>(H.data(), H.rows()*H.cols());
//         
//         M->dispose();
//         return(EXIT_SUCCESS);
// 
//     }
#endif
    
    
    
          /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
        int solveUnconstrained ( double l1, double l2,double prec,double lam )
    {
        
        std::cout << "Kx Matrix 5: " << Kx.full()  << std::endl;
        precomputeKernelMatrices ( l1, l2,prec,lam );
        std::cout << "Kx Matrix 0: " << Kx.full() << std::endl;

        h = (Qy * ( -prob_vec.cwiseQuotient (prob_quadFormMat).reshaped(Qy.cols(), Qx.cols()) ) * Qx.transpose()).reshaped();
        H = h.reshaped(Qy.cols(), Qx.cols());

        return ( EXIT_SUCCESS );
    }
    
              /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
        int solveUnconstrainedMomentMatching ( double l1, double l2,double prec,double lam )
    {
        
        precomputeKernelMatrices ( l1, l2,prec,lam );
        h = (Qy * ( -prob_vec.cwiseQuotient (prob_quadFormMat).reshaped(Qy.cols(), Qx.cols()) ) * Qx.transpose()).reshaped();
        H = h.reshaped(Qy.cols(), Qx.cols());

        return ( EXIT_SUCCESS );
    }
    
    
    /*
   *    \brief this is the L2 loss function to be evaluated at the validation or test data
   */
    double validationScore(const Matrix& Xs, const Matrix& Ys) const{
        const double n(Xs.cols());
        const Matrix& ky = basy.eval(Ys);
        const Matrix& kx = basx.eval(Xs);
        const Matrix oas = ky * H * kx.transpose();
        
        return(((oas.transpose() * oas).trace()+2.0 * oas.sum())/(n*n)-2.0/n*oas.trace());
    }

    
            /*
   *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
   */
    Matrix condExpfVec ( const Matrix& Xs, bool structuralYes = false ) const
    {

        const Matrix res = Kyblock *H*basx.eval(Xs).transpose() + Matrix::Constant(Kyblock.rows(),Xs.cols(),1);
        if(!structuralYes){
            return(res.array().rowwise()/res.colwise().sum().array());
        }
        // const Matrix resres = res.array().rowwise()/res.colwise().sum().array();

        return ( res.array().cwiseMax(0.0).rowwise()/res.cwiseMax(0.0).colwise().sum().array());
    }
    
            /*
    *    \brief computes conditional expectation of all the rows in Ys
    *    revise this TODO 
    */
    const Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs, bool structuralYes = false  ) const {
        Matrix inter = H * basx.eval(Xs).transpose();
        if(!structuralYes){
            const RowVector norma = Kyblock.colwise().sum() * inter + RowVector::Constant(Xs.cols(),Kyblock.rows());
            const Matrix oida = Ys   * Kyblock; // should be small
            inter.array().rowwise() /= norma.array();
            const Matrix part = Ys.rowwise().sum().replicate(1,Xs.cols()).array().rowwise()/norma.array();
            return ( part+oida * inter );
        }
        const RowVector norma = ((Kyblock * inter) + Matrix::Constant(Kyblock.rows(), Xs.cols(),1)).cwiseMax(0.0).colwise().sum();
            // const Matrix oida = Ys   * Kyblock; // should be small
            inter.array().rowwise() /= norma.array();
            const Matrix part = Ys.rowwise().sum().replicate(1,Xs.cols()).array().rowwise()/norma.array();
            return ( part+ Ys   * (Kyblock *inter).cwiseMax(0.0) );
    }
    
    bool positiveOnGrid() const {
        return(static_cast<bool>((Kyblock * H * Kxblock.transpose()).minCoeff()+1.0 >= 0.0));
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

    Matrix Qx; // the basis transformation matrix
    Matrix Qy; // the basis transformation matrix
    
    Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
    Vector prob_vec; // the important ones
    
    
    Matrix C_Under; // the important ones
    Matrix C_Over; // the important ones
    
    

    KernelMatrix Kx;
    KernelMatrixY Ky;

    LowRank pivx;
    LowRankY pivy;
    
    KernelBasis basx;
    KernelBasisY basy;
    
//  spectral basis bounds. in the first row the mins and in the second row the maxs
    Vector xbound_min;
    Vector xbound_max;
    
    Vector ybound_min;
    Vector ybound_max;
    
//     double tol;
    static constexpr double p = 1.0;
    
    /*
   *    \brief computes the kernel basis and tensorizes it
   */ 
    void precomputeKernelMatrices ( double l1, double l2,double prec, double lam )
    {
//         RRCA::IO::Stopwatch sw;
//         sw.tic();
        std::cout << "Kx Matrix precompute4: " << std::endl;
        std::cout << Kx.full() << std::endl;
        Kx.kernel().l = l1;
        Ky.kernel().l = l2;

        std::cout << "Kx Matrix precompute3: " << std::endl;
        std::cout << Kx.full() << std::endl;
        pivx.compute ( Kx,prec);
        std::cout << "Kx Matrix precompute2: " << std::endl;
        std::cout << Kx.full() << std::endl;
        basx.init(Kx, pivx.pivots());
        std::cout << "Kx Matrix precompute1: " << std::endl;
        std::cout << Kx.full() << std::endl;
        basx.initSpectralBasisWeights(pivx);
        
        std::cout << "Kx Matrix precompute --: " << std::endl;
        std::cout << Kx.full() << std::endl;
        
        Qx =  basx.matrixQ() ;
        Kxblock = basx.eval(xdata);
        

        LX = Kxblock * Qx;
        
        
        pivy.compute ( Ky,prec);
        basy.init(Ky, pivy.pivots());
        basy.initSpectralBasisWeights(pivy);
        
        // std::cout << "Kx Matrix precompute2: " << std::endl;
        // std::cout << Kx.full() << std::endl;

        
        Qy =  basy.matrixQ() ;
        Kyblock = basy.eval(ydata);
        

        LY = Kyblock * Qy;
        
        xbound_min = LX.colwise().minCoeff().transpose();
        xbound_max = LX.colwise().maxCoeff().transpose();
        
        
        ybound_min = LY.colwise().minCoeff().transpose();
        ybound_max = LY.colwise().maxCoeff().transpose();
        
        
//      bounds
        C_Under = ybound_min.cwiseMax(0) * xbound_min.cwiseMax(0).transpose()+ybound_max.cwiseMin(0).cwiseAbs() * xbound_max.cwiseMin(0).cwiseAbs().transpose() 
                - (ybound_max.cwiseMax(0) * xbound_min.cwiseMin(0).cwiseAbs().transpose()).cwiseMax(ybound_min.cwiseMin(0).cwiseAbs() * xbound_max.cwiseMax(0).transpose());
                
        C_Over = (ybound_max.cwiseMax(0) * xbound_max.cwiseMax(0).transpose()).cwiseMax(ybound_min.cwiseMin(0).cwiseAbs() * xbound_min.cwiseMin(0).cwiseAbs().transpose()) 
                - ybound_max.cwiseMin(0).cwiseAbs() * xbound_min.cwiseMax(0).transpose()-ybound_min.cwiseMax(0) * xbound_max.cwiseMin(0).cwiseAbs().transpose();
        
        
        

        precomputeHelper(lam);
        
        // std::cout << " passed by precompute " << LX.cols() << '\t' << LY.cols() << std::endl;

    }
    

    
    
    
    
    void precomputeHelper(double lam){
        const unsigned int n ( LX.rows() );
        const unsigned int  rankx ( LX.cols() );
        const unsigned int  ranky ( LY.cols());
        const unsigned int  m ( rankx*ranky );
       
        Vector p_ex_k = (LY.transpose() * LX).reshaped();
        Vector px_o_py_k = (LY.colwise().mean().transpose() * LX.colwise().mean()).reshaped()*n;

        Xvar =  ( LX.transpose() * LX).diagonal()  ;
        Yvar =  ( LY.transpose() * LY).diagonal() ;

        prob_quadFormMat= (Yvar * Xvar.transpose()).reshaped().array()/static_cast<double> ( n )+n*lam;
        prob_vec = px_o_py_k-p_ex_k;
    }
#ifdef RRCA_HAVE_MOSEK
    int solveMosek(bool pointWisePositivity = false){

        M_Model M = new mosek::fusion::Model ( "DistributionEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
        

        M->setSolverParam("numThreads", 4);


        const unsigned int m ( Kyblock.cols() * Kxblock.cols() );
        
        
       
//         can define matrix variable for the bilinear form. this is actually the transpose of HH, because mosek is row major
        M_Variable::t HH = M->variable("H", monty::new_array_ptr<int, 1>({(int)LX.cols(), (int)LY.cols()}), M_Domain::unbounded());
        M_Variable::t HH_m = M->variable("Hm", monty::new_array_ptr<int, 1>({(int)LX.cols(), (int)LY.cols()}), M_Domain::greaterThan(0.0));
        M_Variable::t HH_p = M->variable("Hp", monty::new_array_ptr<int, 1>({(int)LX.cols(), (int)LY.cols()}), M_Domain::greaterThan(0.0));
            
        M_Variable::t ht = M_Var::flatten( HH );
        
//         for the quadratic cone
        M_Variable::t uu = M->variable( "uu",1, M_Domain::greaterThan(0.0));

         auto quadsqrt = std::make_shared<M_ndarray_1>( monty::shape ( m ), [&] ( ptrdiff_t l ) { return sqrt ( prob_quadFormMat  ( l )); } );
        auto prob_vecwrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( prob_vec.data(), monty::shape ( m) )) ;
        
        M->constraint("oida", M_Expr::vstack(0.5, uu, M_Expr::mulElm(quadsqrt,ht)), M_Domain::inRotatedQCone()); // quadratic cone for objective function
        

        const unsigned int n ( LY.rows() );
        Matrix Amat_t ( m,n );

//         deal first with E[g|x_i] = 1, i=1,...,n
//         Vector ysums ( LY.colwise().mean());

        //         now project onto \psi X
         Vector LYmeans(LY.colwise().mean());
         Vector LXmeans(LX.colwise().mean());
         
         auto LYmeanswrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( LYmeans.data(), monty::shape ( LYmeans.size()) ) );
         auto LXmeanswrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( LXmeans.data(), monty::shape ( LXmeans.size()) ) );
         
         
        
//      normalization constraint
        M->constraint(M_Expr::dot(LXmeanswrap,M_Expr::mul(HH,LYmeanswrap)), M_Domain::equalsTo(0.0)); 
//         poitivity constraints
//         std::cout << "kurz vorher " << std::endl;

        if(pointWisePositivity){
            const M_Matrix::t Qxwrap_t = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( Qx.data(), monty::shape ( Qx.cols(), Qx.rows()) ) ));
            const M_Matrix::t Qywrap_t = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2 ( Qy.data(), monty::shape ( Qy.cols(), Qy.rows()) ) ));
            M->constraint(M_Expr::sub(M_Expr::mul(Qxwrap_t->transpose(),M_Expr::mul(HH,Qywrap_t)), M_Expr::sub(HH_p,HH_m)), M_Domain::equalsTo(0.0));
            M->constraint ( M_Expr::sum(HH_m),M_Domain::lessThan ( p ) ); 
//             std::cout << "arrived hier " << std::endl;
        } else {
//        compute the kernel functions with the min distance vectors
//             Matrix LXsub = LX.topRows(std::min(static_cast<Eigen::Index>(RRCA_LOWRANK_STEPLIM),xdata.cols()));
//             Matrix LYsub = LY.topRows(std::min(static_cast<Eigen::Index>(RRCA_LOWRANK_STEPLIM),xdata.cols()));
// //          compute only for  
//             const M_Matrix::t Lxwrap_t = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( LXsub.data(), monty::shape ( LXsub.cols(), LXsub.rows()) ) ));
//             const M_Matrix::t Lywrap_t = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( LYsub.data(), monty::shape ( LYsub.cols(), LYsub.rows()) ) ));
            // M->constraint ( M_Expr::mul(Lxwrap_t->transpose(),M_Expr::mul(HH, Lywrap_t)),M_Domain::greaterThan ( - p ) );
            // auto xbound_min_wrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( xbound_min.data(), monty::shape ( xbound_min.size()) ) );
            // auto xbound_max_wrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( xbound_max.data(), monty::shape ( xbound_max.size()) ) );
            // auto ybound_min_wrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( ybound_min.data(), monty::shape ( ybound_min.size()) ) );
            // auto ybound_max_wrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( ybound_max.data(), monty::shape ( ybound_max.size()) ) );
            M->constraint(M_Expr::sub(M_Expr::sub(HH_p,HH_m), HH), M_Domain::equalsTo(0.0));
            
            const M_Matrix::t C_underwrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( C_Under.data(), monty::shape ( C_Under.cols(), C_Under.rows()) ) ));
            const M_Matrix::t C_overwrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( C_Over.data(), monty::shape ( C_Over.cols(), C_Over.rows()) ) ));
            M->constraint(M_Expr::sub(M_Expr::dot(C_underwrap,HH_p),M_Expr::dot(C_overwrap,HH_m)),M_Domain::greaterThan ( - p ));
            
            // M->constraint ( M_Expr::dot(xbound_min_wrap,M_Expr::mul(HH, ybound_min_wrap)),M_Domain::greaterThan ( - p ) );
            // M->constraint ( M_Expr::dot(xbound_min_wrap,M_Expr::mul(HH, ybound_max_wrap)),M_Domain::greaterThan ( - p ) );
            // M->constraint ( M_Expr::dot(xbound_max_wrap,M_Expr::mul(HH, ybound_min_wrap)),M_Domain::greaterThan ( - p ) );
            // M->constraint ( M_Expr::dot(xbound_max_wrap,M_Expr::mul(HH, ybound_max_wrap)),M_Domain::greaterThan ( - p ) );
        }
        //         !!!!!!!!!
        
        M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(uu,M_Expr::mul ( 2.0,M_Expr::dot ( prob_vecwrap, ht ) )));
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
//             const auto htsol = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( * ( ht->level() ) ) );
            M_ndarray_1 htsol   = * ( HH->level() );
            const Eigen::Map<Matrix> auxmat ( htsol.raw(), LY.cols() ,LX.cols() );
            H = Qy * auxmat * Qx.transpose();
            h = H.reshaped();
        } else {
            std::cout << "infeasible  " <<  std::endl; 
            M->dispose();
            return ( EXIT_FAILURE );
        }
        M->dispose();
        return ( EXIT_SUCCESS );
    }
#endif
// #ifdef RRCA_HAVE_GUROBI
//     int solveGurobi(){
//         std::unique_ptr<GRBEnv> env =  make_unique<GRBEnv>();
//         GRBModel model = GRBModel ( *env );
//         const size_t m ( Kyblock.cols() * Kxblock.cols() );
//         
// 
//         Vector lb ( Vector::Constant ( m,-100.0 ) );
//         Vector ub ( Vector::Constant ( m,100.0 ) );
//         
// 
//         std::unique_ptr<GRBVar[]> ht(model.addVars ( lb.data(), ub.data(), NULL, NULL, NULL,  m ));
//         std::unique_ptr<GRBVar[]>  hp(model.addVars ( NULL, ub.data(), NULL, NULL, NULL,  m ));
//         std::unique_ptr<GRBVar[]>  hm(model.addVars ( NULL, ub.data(), NULL, NULL, NULL,  m ));
// 
// 
//         GRBQuadM_Expr qobj ( 0 );
//         GRBLinM_Expr lobj1 ( 0 );
//         GRBLinM_Expr lobj2 ( 0 );
//         GRBLinM_Expr lobj3 ( 0 );
//         GRBLinM_Expr lobj4 ( 0 );
// //         const double fxsup(Kx.kernel().sup);
// //         const double fysup(Ky.kernel().sup); 
//         
//         for ( unsigned int i = 0; i < m; ++i ) {
//             qobj    += prob_quadFormMat ( i ) * ht[i] * ht[i]; //variance quadratic form plus regularization
//             lobj1   += 2.0*prob_vec ( i ) * ht[i]; //linear part
//             model.addConstr ( ht[i], GRB_EQUAL, hp[i]-hm[i]);
//             lobj2 +=  hp[i]*aa(i)-hm[i]*bb(i);
//         }
// 
//         model.addConstr ( lobj2, GRB_LESS_EQUAL,p );
//         
//         GRBQuadM_Expr obj ( qobj + lobj1 );
//         model.setObjective ( obj );
// 
//         const unsigned int n ( LY.rows() );
//         Matrix Amat_t ( m,n );
// 
// //         deal first with E[g|x_i] = 1, i=1,...,n
//         Vector ysums ( LY.colwise().mean());
// 
//         //         now project onto \psi X
// //      first compute projection matrix
//         const size_t mX ( LX.cols() );
//         const size_t mY ( LY.cols() );
//         const Vector LYmeans(LY.colwise().mean());
//         const Vector LXmeans(LX.colwise().mean());
//         
//         for ( auto i = 0; i < mX; ++i ) {
//             lobj1 = 0;
//             for ( auto j = 0; j < mY; ++j ) {
//                 lobj1 += ht[i*mY + j] * LYmeans(j);
//             }
//             model.addConstr ( lobj1, GRB_EQUAL,0 );
//         }
//         
// 
//         for ( auto i = 0; i < mY; ++i ) {
//             lobj1 = 0;
//             for ( auto j = 0; j < mX; ++j ) {
//                 lobj1 += ht[i + j*mY] * LXmeans(j);
//             }
//             model.addConstr ( lobj1, GRB_EQUAL,0);
//         }
// 
//         model.optimize();
//         if ( model.get ( GRB_IntAttr_Status ) == GRB_OPTIMAL) {
//             Vector aux ( m );
//             for ( size_t i = 0; i < m; ++i ) {
//                 aux ( i ) = ht[i].get ( GRB_DoubleAttr_X );
//             }
// //             h= Q*aux;
//             Eigen::Map<Matrix> auxmat ( aux.data(), LY.cols() ,LX.cols() );
//             H = Qy * auxmat * Qx.transpose();
//             h = Eigen::Map<Vector>(H.data(), H.rows()*H.cols());
// //             std::cout << h.transpose() << std::endl;
//         } else {
//             std::cout << "infeasible  " <<  std::endl; 
//             return ( EXIT_FAILURE );
//         }
//         return ( EXIT_SUCCESS );
//     }
// #endif
};


/*
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
constexpr double DistributionEmbedding<KernelMatrix, LowRank, KernelBasis,KernelMatrixY, LowRankY, KernelBasisY>::p;*/
















} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
