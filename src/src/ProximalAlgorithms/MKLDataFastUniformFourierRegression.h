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
#ifndef RRCA_MKLDATAFASTUNIFORMFOURIERREGRESSION_H_
#define RRCA_MKLDATAFASTUNIFORMFOURIERREGRESSION_H_



namespace RRCA {
    

class MKLDataFastUniformFourierRegression {
    
    public:
    /*
    *    \brief Minimizes 1/(2n)\|y - f(X)\|_2^2 +\lambda/2 \|f\|^2_H, where H is a MKL hypothesis space. here in terms
    * of fourier series. 
    * this is a fast implementation in terms of rank one matrices
   */
        MKLDataFastUniformFourierRegression(const Vector &y_, const Matrix& X_ ) :
        y(y_),
        X(X_),
        u(M_PI * Vector::Random(X_.rows()).array()+M_PI),
        n(X_.rows()),
        m(X_.rows()) {
            assert(y_.size() == n);
            Matrix covar = X_ * X_.transpose(); // this should be n x n
            G = ((2.0 * M_PI * covar).rowwise()+u.transpose()).array().cos()/(sqrt(M_PI));

            
        }
        int solve(double lambda,  double rho = 1.0){
            alpha = Vector::Constant(m, 1.0/static_cast<double>(m));
            double const noverrho(static_cast<double>(n)/rho);
            const double nd(static_cast<double>(n));
            Matrix sepp;
            
            Ghat = G.array().rowwise()*alpha.array().transpose();
            sepp = Ghat.transpose()*Ghat;
            sepp.diagonal() += nd * lambda * alpha;
            ctil = sepp.ldlt().solve(Ghat.transpose()*y);
            double eps1(1);
            double eps2(2);
            unsigned int counter(1);
            Vector beta_1   = alpha;
            Vector gamma_1  = alpha;
            Vector delta_1  = Vector::Constant(m, 0);
            
            Vector beta_0;
            Vector gamma_0;
            Vector delta_0;
            
             // const Eigen::ArrayXi ind = Eigen::ArrayXi::LinSpaced(m,0,m-1);
            
            while(counter < 20000 && std::max(eps1, eps2)> 1.e-4){
                beta_0 = beta_1;
                gamma_0 = gamma_1;
                delta_0 = delta_1;
                
                Gtil = G.array().rowwise()*ctil.array().transpose(); // this is n x m
                // std::cout << Gtil.upperLeftCorner(5,5) << std::endl;
                
                beta_1  =  (2.0*Gtil.transpose() * Gtil + noverrho*Matrix::Identity(m,m)).llt().solve(2.0*Gtil.transpose() * y-nd*lambda*ctil.cwiseProduct(ctil)+noverrho * (gamma_0-delta_0));
                gamma_1 = simplex_project(beta_1+delta_0);
                delta_1 = delta_0 + beta_1-gamma_1;
                
                std::vector<int> ind;
                for(auto j = 0; j < m; ++j){
                    if(gamma_1(j)>0.0) ind.push_back(j);
                }
                
                // Ghat = G.array().rowwise()*gamma_1.array().transpose();
                // sepp = Ghat.transpose()*Ghat;
                // sepp.diagonal() += nd * lambda * gamma_1;
                // ctil = sepp.ldlt().solve(Ghat.transpose()*y);
                ctil.setZero();
                Ghat = G(Eigen::all,ind).array().rowwise()*gamma_1(ind).array().transpose();
                sepp = Ghat.transpose()*Ghat;
                sepp.diagonal() += nd * lambda * gamma_1(ind);
                Vector res = sepp.llt().solve(Ghat.transpose()*y);
                int counter2 = 0; 
                for(auto j = 0; j < ind.size(); ++j){
                    ctil(ind[j]) = res(counter2);
                    ++counter2;
                }
                
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
       const Vector& getCtil() const {return ctil;}
       const Vector& getMosekCtil() const {return mosek_ctil;}
       double getObj() const {
            return((y-G * alpha.asDiagonal() * ctil).squaredNorm());
       }
       
       double getMosekObj() const {
           Matrix mat(n,2);
           mat << y , G * mosek_ctil;
           // std::cout << mat << std::endl;
            return((y-G * mosek_ctil).squaredNorm());
       }
       
       int solveMosek(double lambda){
           M_Model M = new mosek::fusion::Model ( "MKLDataFastUniformFourierRegression" );
        // M->setLogHandler ( [=] ( const std::string & msg ) {
        //     std::cout << msg << std::flush;
        // } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
        
        M_Variable::t b_t = M->variable("b_t", m, M_Domain::unbounded());
        M_Variable::t a_t = M->variable("c_t", m, M_Domain::greaterThan(0.0));
        M_Variable::t uu = M->variable("uu", M_Domain::unbounded());
        M_Variable::t vv = M->variable("vv",m, M_Domain::unbounded());
        
        const M_Matrix::t Q_twrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2> (new  M_ndarray_2( G.data(), monty::shape ( G.cols(), G.rows()) ) ));
        Vector y_nonconst = y;
        auto y_wrap =  std::shared_ptr<M_ndarray_1> (new M_ndarray_1( y_nonconst.data(), monty::shape ( y_nonconst.size()) ) );
        
        
        const double nd(static_cast<double>(n));
        M->constraint(M_Expr::vstack(0.5, uu, M_Expr::sub(y_wrap,M_Expr::mul(Q_twrap->transpose(),b_t ))), M_Domain::inRotatedQCone()); // quadratic cone for objective function
//         now add many additional rotated quadratic cones
        M->constraint(M_Expr::hstack(M_Expr::mul(vv,0.5),a_t , b_t), M_Domain::inRotatedQCone());
        M->constraint(M_Expr::sum(a_t), M_Domain::equalsTo(1.0));
        
        M->objective (  mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(M_Expr::mul(1.0/nd,uu),M_Expr::mul(lambda,M_Expr::sum (vv))));
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
            M_ndarray_1 csol   = * ( b_t->level() );
            mosek_ctil = Eigen::Map<Vector>( csol.raw(), m );
            
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
        const Vector u;
        
        
        const unsigned int n; // how many data points do we have
        const unsigned int m; // how large is the dimension of X
        
        
        
        Vector alpha; // Vector of MKL weights
        Vector ctil; // this is G'c
        Vector mosek_ctil; // this is AG'c
        Matrix G;
        Matrix G_t;
        Matrix Ghat;
        Matrix Gtil;
        
       
        

        

    
    };
    
    
    
    
    
     
    
} // RRCA

#endif
