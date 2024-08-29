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
#ifndef RRCA_SCENARIOANALYSIS_TOTALVARIATIONSCENARIOS_H_
#define RRCA_SCENARIOANALYSIS_TOTALVARIATIONSCENARIOS_H_

// #include <eigen3/Eigen/Dense>
namespace RRCA
{
namespace SCENARIOANALYSIS
{
   
    
    
    
    class TotalVariationScenarios {


    
    public:
    /*
    *    \brief Given moment matrix A and empirical scenarios pts computes least squares probabilities that fit to A 
   */
        TotalVariationScenarios(const Matrix& pts_, unsigned int order_, double tol_= RRCA_LOWRANK_EPS) : 
        pts(pts_), 
        order(order_), 
        datadim(pts_.rows()), // assuming that data is in col major order (time series along the x-axis
        n(pts_.cols()),
        basisdim(binomialCoefficient ( order_+ pts_.rows(), pts_.rows()  )),
        tol(tol_),
        myIndex ( pts_.rows(),order_  ),
        bench(pts_.cols()),
        V(basisdim,n) {
            const auto &mySet = myIndex.get_MultiIndexSet();
//          copy into vector for simplicity
            unsigned int counter;
//             fill V matrix
            for(unsigned int i = 0; i < n; ++i){
                counter = 0;
                double accum(1);
                for ( const auto &ind1 : mySet ) {
                    accum = 1;
                    for(unsigned int j = 0; j < datadim; ++j){
                        accum *= pow(pts(j,i),ind1(j));
                    }
                    V(counter, i) = accum;
                    ++counter;
                }
//              compute the matrix 1 norm
                bench(i) = (V.col(i) * V.col(i).transpose()).cwiseAbs().colwise().sum().maxCoeff();
            }
//             std::cout << "bench:\n" << bench << std::endl;
        }
        int solve(){
            A = (V * V.transpose()/static_cast<double>(pts.cols()));
            A(0,0) = 1;
            const double one_n(1.0/static_cast<double>(n));

            M_Model M = new mosek::fusion::Model ( "TotalVariationScenarios" );
            M->setLogHandler ( [=] ( const std::string & msg ) {
                std::cout << msg << std::flush;
            } );
            auto _M = monty::finally ( [&]() {
                M->dispose();
            } );
            M_Variable::t P = M->variable ( "P",n,M_Domain::inRange(-one_n, 1.0-one_n));
            
            M_Variable::t Pp_til = M->variable ( "Pp_til",n,M_Domain::inRange(0.0, 1.0-one_n));
            M_Variable::t Pm_til = M->variable ( "Pm_til",n,M_Domain::inRange(0.0, one_n));
            
//          indicator is one if Pp is on and zero otherwise
            M_Variable::t flags = M->variable ( "Indicator",n,M_Domain::binary());
            M_Variable::t Pp = M->variable ( "Pp",n,M_Domain::inRange(0.0, 1.0-one_n));
            M_Variable::t Pm = M->variable ( "Pm",n,M_Domain::inRange(0.0, one_n));
            M_Variable::t u = M->variable ( "u",1,M_Domain::greaterThan(0.0));
            
            
            
//             linearize the bilinear form involving the indicator and the tilde variables
//             first the plus variable
            M->constraint(M_Expr::sub(Pp,M_Expr::mul(1.0-one_n,flags)), M_Domain::lessThan(0.0));
            M->constraint(M_Expr::sub(Pp,Pp_til), M_Domain::lessThan(0.0));
            M->constraint(M_Expr::sub(M_Expr::sub(Pp,Pp_til),M_Expr::mul(1.0-one_n,flags)), M_Domain::greaterThan(-(1.0-one_n)));
            
            //             now the minus variabe
            M->constraint(M_Expr::add(Pm,M_Expr::mul(one_n,flags)), M_Domain::lessThan(one_n));
            M->constraint(M_Expr::sub(Pm,Pm_til), M_Domain::lessThan(one_n));
            M->constraint(M_Expr::add(M_Expr::sub(Pm,Pm_til),M_Expr::mul(one_n,flags)), M_Domain::greaterThan(0.0));
            
            auto benchWrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( bench.data(), monty::shape ( n) )) ;
            M->constraint(M_Expr::sum(P), M_Domain::equalsTo(0.0));
            M->constraint(M_Expr::sub(P,M_Expr::sub(Pp,Pm)), M_Domain::equalsTo(0.0));
            
//             M->constraint(M_Expr::dot(M_Expr::add(Pp,Pm),benchWrap), M_Domain::lessThan(tol));
            const M_Matrix::t V_wrap_t = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2  ( V.data(), monty::shape ( V.cols(), V.rows() ) ) ) );
            M->constraint(M_Expr::vstack(u, M_Expr::flatten(M_Expr::mul(V_wrap_t->transpose(),M_Expr::mulElm(M_Var::hrepeat(P,V.rows()), V_wrap_t)))), M_Domain::inQCone());
            M->constraint(u, M_Domain::lessThan(tol));
            
            
            M->objective ( mosek::fusion::ObjectiveSense::Maximize, M_Expr::sum(Pm));
        
            M->solve();
            if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal || M->getDualSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
                auto resMat = * ( P->level() );
//              now find the indices with non zero probability. since we are using a linear program, the probs will be exactly zero
                Vector inter = Eigen::Map<Vector>(resMat.raw(),n).array() + one_n;
                std::cout << inter.transpose() << std::endl;
                std::cout << "Pm:\n " << Eigen::Map<Vector>((* ( Pm->level() )).raw(),n).transpose() << std::endl;
                std::cout << "Pp:\n " << Eigen::Map<Vector>((* ( Pp->level() )).raw(),n).transpose() << std::endl;
                
                Matrix pmpp(n,2);
                pmpp << Eigen::Map<Vector>((* ( Pm->level() )).raw(),n).array(), Eigen::Map<Vector>((* ( Pp->level() )).raw(),n).array();
                std::cout << "pmpp: \n" << pmpp << std::endl;
                
                
                std::vector<Index> indi;
                
                for(Index i = 0; i < n; ++i){
                    if(fabs(inter(i)) > 1e-5){
                        indi.push_back(i);
                    }
                }
                resP = inter(indi);
                resP/=resP.sum();
                scen = pts(Eigen::all, indi);
                
                std::cout << "resP:\n " << resP << std::endl;
                std::cout << "scen:\n " << scen << std::endl;

//                 mosek::releaseGlobalEnv();
            } else {
                std::cout << "least squares infeasible " << std::endl;
                return(EXIT_FAILURE);
            }

            
            
            return(EXIT_SUCCESS);
        }
        Vector getProbs() const {
            return(resP);
        }
        Matrix getScenarios() const {
            return(scen);
        }
        void setTol(double tol_){
            tol = tol_;
        }
        double getTol() const {return(tol);}
        
        const Matrix& getA() const {return(A);}
        
        Matrix getScenA(const Vector& PP){
            const auto &mySet = myIndex.get_MultiIndexSet();
//          copy into vector for simplicity
            unsigned int counter;
            Matrix scenV(basisdim,scen.cols());
//             fill V matrix
            for(unsigned int i = 0; i < scen.cols(); ++i){
                counter = 0;
                double accum(1);
                for ( const auto &ind1 : mySet ) {
                    accum = 1;
                    for(unsigned int j = 0; j < datadim; ++j){
                        accum *= pow(scen(j,i),ind1(j));
                    }
                    scenV(counter, i) = accum;
                    ++counter;
                }
            }

            return(scenV * PP.asDiagonal() * scenV.transpose());
        }

        
    private:
        const Matrix &pts;
        const unsigned int order; // what is the order
        const unsigned int datadim; // how many variables
        const unsigned int n; // how many data points do we have
        const unsigned int basisdim; // how many monomials do we have in the monomial basis
        double tol;
        
        const RRCA::MultiIndexSet<iVector> myIndex;
        
        Vector bench;
        
        
        
        Matrix A;
        Matrix V;
        Matrix scen;
        Vector resP;
    
    };
    
  
    
    
    
    
    
     
    
} // SCENARIOANALYSIS
} // RRCA

#endif
