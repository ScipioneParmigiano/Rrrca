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
#ifndef RRCA_MKLLOWRANKDATAFASTFOURIERREGRESSION_H_
#define RRCA_MKLLOWRANKDATAFASTFOURIERREGRESSION_H_

#include "../../../RRCA/ScenarioAnalysis"


namespace RRCA {
    

class MKLLowRankDataFastFourierRegression {
    
    public:
    /*
    *    \brief Minimizes 1/(2n)\|y - f(X)\|_2^2 +\lambda/2 \|f\|^2_H, where H is a MKL hypothesis space. here in terms
    * of fourier series. 
    * this is a fast implementation in terms of rank one matrices
   */
        MKLLowRankDataFastFourierRegression(const Vector &y_, const Matrix& X_ ) :
        y(y_),
        X(X_),
        n(X_.rows()),
        dim(X_.cols()) {
            assert(y_.size() == n);
                unsigned int order = 1;
                unsigned int basisdim = binomialCoefficient ( 2* order + X_.cols(), X_.cols()  );
                std::cout << " basis dim " << basisdim << std::endl;
                if(basisdim <=n){
                    while(basisdim < n) {
                        ++order;
                        basisdim = binomialCoefficient ( 2* order + X_.cols(), X_.cols()  );
                    }
                    basisdim = binomialCoefficient ( 2* (order-1) + X_.cols(), X_.cols()  );
                
                    RRCA::SCENARIOANALYSIS::OMPScenarios ompScen(X.transpose(), (order-1), 1e-12);
                    ompScen.solve();
                    scenX = ompScen.getScenarios().transpose();
                    std::cout << " chose OMP scenarios with  " << scenX.rows() << " scenarios for order " << order-1 << std::endl;
                } else {
                    RRCA::SCENARIOANALYSIS::CovarianceScenarios covarscen(X.transpose());
                    covarscen.compute();
                    scenX = covarscen.getScenarios().transpose();
                    std::cout << " chose covariance scenarios " <<std::endl;
                }
                

            
            m = scenX.rows();
            
            
            
            
            Matrix covar = X_ * scenX.transpose(); // this should be n x m
//          V and W now are tall
            V = (2.0 * M_PI * covar).array().cos();
            W = (2.0 * M_PI * covar).array().sin();

            
        }
        int solve(double lambda,  double rho = 1.0){
            alpha = Vector::Constant(m, 1.0/static_cast<double>(m));
            double const noverrho(static_cast<double>(n)/rho);
            const double nd(static_cast<double>(n));
//          compute coefficient vector delta=[\beta,\gamma]' for the first time
            Matrix UA(n,2*m);
            // 
            UA.leftCols(m) = V.array().rowwise() * alpha.array().transpose();
            UA.rightCols(m) = W.array().rowwise() * alpha.array().transpose();
            Matrix inter = UA.transpose() * UA;
            inter.diagonal() += nd*lambda*alpha.replicate(2,1);
            Vector betgam = inter.ldlt().solve(UA.transpose() * y);
            // std::cout << "betgam " << betgam.transpose() << std::endl;
            // Z = V * alpha.asDiagonal() * V.transpose() + W * alpha.asDiagonal() * W.transpose();
            // c = (Z+nd * lambda * Matrix::Identity(m,m)).llt().solve(y);
            double eps1(1);
            double eps2(2);
            unsigned int counter(1);
            Vector beta_1   = alpha;
            Vector gamma_1  = alpha;
            Vector delta_1  = Vector::Constant(m, 0);
            
            Vector beta_0;
            Vector gamma_0;
            Vector delta_0;
            
             
            
            while(counter < 20000 && std::max(eps1, eps2)> 1.e-4){
                beta_0 = beta_1;
                gamma_0 = gamma_1;
                delta_0 = delta_1;
                
                // bet = V.transpose() * c;
                // gam = W.transpose() * c;
                
                bet = betgam.head(m);
                gam = betgam.tail(m);
                
                H = V.array().rowwise() * bet.array().transpose() + W.array().rowwise() * gam.array().transpose(); // this is n x m
                
                beta_1  =  (2.0*H.transpose() * H + noverrho*Matrix::Identity(m,m)).llt().solve(2.0*H.transpose() * y-nd*lambda*(bet.cwiseProduct(bet)+gam.cwiseProduct(gam))+noverrho * (gamma_0-delta_0));
                gamma_1 = simplex_project(beta_1+delta_0);
                delta_1 = delta_0 + beta_1-gamma_1;
                
                // Z = V * gamma_1.asDiagonal() * V.transpose() + W * gamma_1.asDiagonal() * W.transpose();
                // c = (Z+nd * lambda * Matrix::Identity(m,m)).llt().solve(y);
                
                
                UA.leftCols(m) = V.array().rowwise() * gamma_1.array().transpose();
                UA.rightCols(m) = W.array().rowwise() * gamma_1.array().transpose();
                inter = UA.transpose() * UA;
                inter.diagonal() += nd*lambda*gamma_1.replicate(2,1);
                betgam = inter.ldlt().solve(UA.transpose() * y   );
                
                eps1 = (beta_1-beta_0).norm();
                eps2 = (delta_1-delta_0).norm();
                // std::cout << eps1 << '\t' << eps2 << std::endl;
                ++counter;
            }
            alpha = gamma_1;
            std::cout << "counter " << counter << std::endl;
            return(EXIT_SUCCESS);
        }
       const Vector& getAlpha() const {return alpha;} 
       const Vector& getBeta() const {return bet;}
       const Vector& getGamma() const {return gam;}
       const Vector& getC() const {return c;}
       double getObj() const {
            return((y-((V.array().rowwise()*alpha.array().transpose()).matrix() * bet + (W.array().rowwise()*alpha.array().transpose()).matrix() * gam)).squaredNorm());
       }
       
       double getMosekObj() const {
            return((y-(V * mosek_bet + W *  mosek_gam)).squaredNorm());
       }
       
       
       int solveMosek(double lambda){
           M_Model M = new mosek::fusion::Model ( "MKLLowRankDataFastFourierRegression" );
        // M->setLogHandler ( [=] ( const std::string & msg ) {
        //     std::cout << msg << std::flush;
        // } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
        
        M_Variable::t bet_t = M->variable("bet_t", m, M_Domain::unbounded());
        M_Variable::t gam_t = M->variable("gam_t", m, M_Domain::unbounded());
        M_Variable::t a_t = M->variable("c_t", m, M_Domain::greaterThan(0.0));
        M_Variable::t uu = M->variable("uu", M_Domain::unbounded());
        M_Variable::t vv_b = M->variable("vv_b",m, M_Domain::unbounded());
        M_Variable::t vv_g = M->variable("vv_g",m, M_Domain::unbounded());
        
        const M_Matrix::t V_twrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( V.data(), monty::shape ( V.cols(), V.rows()) ) ));
        const M_Matrix::t W_twrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( W.data(), monty::shape ( W.cols(), W.rows()) ) ));
        Vector y_nonconst = y;
        auto y_wrap =  std::shared_ptr<M_ndarray_1> (new M_ndarray_1( y_nonconst.data(), monty::shape ( y_nonconst.size()) ) );
        
        
        const double nd(static_cast<double>(n));
        M->constraint(M_Expr::vstack(0.5, uu, M_Expr::sub(y_wrap,M_Expr::add(M_Expr::mul(V_twrap->transpose(),bet_t ),M_Expr::mul(W_twrap->transpose(),gam_t )))), M_Domain::inRotatedQCone()); // quadratic cone for objective function
//         now add many additional rotated quadratic cones
        M->constraint(M_Expr::hstack(M_Expr::mul(vv_b,0.5),a_t , bet_t), M_Domain::inRotatedQCone());
        M->constraint(M_Expr::hstack(M_Expr::mul(vv_g,0.5),a_t , gam_t), M_Domain::inRotatedQCone());
        M->constraint(M_Expr::sum(a_t), M_Domain::equalsTo(1.0));
        
        M->objective (  mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(M_Expr::mul(1.0/nd,uu),M_Expr::mul(lambda,M_Expr::add(M_Expr::sum(vv_b), M_Expr::sum (vv_g)))));
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
            M_ndarray_1 bet_sol   = * ( bet_t->level() );
            mosek_bet = Eigen::Map<Vector>( bet_sol.raw(), m );
            M_ndarray_1 gam_sol   = * ( gam_t->level() );
            mosek_gam = Eigen::Map<Vector>( gam_sol.raw(), m );
            
            M_ndarray_1 asol   = * ( a_t->level() );
            alpha = Eigen::Map<Vector>( asol.raw(), m ).unaryExpr([](double z){return (z>1e-8?z:0.0);});
            alpha/=alpha.sum();
            // for(auto i = 0; i < m; ++i){
            //     if(alpha(i) > 0.0){
            //         ctil(i)/=alpha(i);
            //         ctil(i) = 0;
            //     }
            // }
        } else {
            std::cout << "infeasible  " <<  std::endl; 
            return ( EXIT_FAILURE );
        }
        
        return(EXIT_SUCCESS);
       }
       
       
        

    private:
        const Vector y;
        const Matrix X;
        Matrix scenX;
        
        
        const unsigned int n; // how many data points do we have
        const unsigned int dim;
        unsigned int m; // how large is the dimension of X
        
        
        
        Vector alpha; // Vector of MKL weights
        Vector bet; //coefficient vector
        Vector gam; //coefficient vector
        Vector mosek_bet; //coefficient vector
        Vector mosek_gam; //coefficient vector
        Vector c;
        
        Matrix V; // this will hold the cos part of cos(2 pi x_i (x-y))=cos(2 pi x_i x)cos(2 pi x_i y)+sin(2 pi x_i x)sin(2 pi x_i y)
        Matrix W; // this will hold the sin part of the above
        Matrix H; // this holds \bs H\isdef [\bm v_1\beta _1 +\bm w_1\gamma _1,\ldots, \bm v_n\beta _n +\bm w_n\gamma _n]
        Matrix Z; // V A V'+W A W'
        

        

    
    };
    
    
    
    
    
     
    
} // RRCA

#endif
