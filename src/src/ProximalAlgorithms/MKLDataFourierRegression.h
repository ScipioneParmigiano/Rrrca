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
#ifndef RRCA_MKLDATAFOURIERREGRESSION_H_
#define RRCA_MKLDATAFOURIERREGRESSION_H_



namespace RRCA {
    

class MKLDataFourierRegression {
    
    public:
    /*
    *    \brief Minimizes \|y - f(X)\|_2^2 +\|f\|^2_H, where H is an MKL hypothesis space, where each dimension of X gets its own kernel
   */
        MKLDataFourierRegression(const Vector &y_, const Matrix& X_ ) :
        y(y_),
        X(X_),
        n(X_.rows()),
        m(X_.rows()),
        Xt(X_.transpose()),
        Kx(m) {
            assert(y_.size() == n);
            
            for ( unsigned int i = 0; i < m; ++i ) {
                Matrix oida(n,n);
                for ( unsigned int k = 0; k < n; ++k ) {
                    for ( unsigned int l = 0; l < n; ++l ) {
                        oida(l,k) = cos(2.0*M_PI*Xt.col(i).dot(Xt.col(l)-Xt.col(k))); 
                    }
                }
                Kx[i] = oida;
                // std::cout << oida.selfadjointView<Eigen::Lower>() << std::endl;
            }
        }
        int solve(double lambda,  double rho = 1){
        // int solve(double lambda, double l, double prec){
            // std::set<unsigned int> indexSet;
            // for ( unsigned int k = 0; k < m; ++k ) {
            //     Kx[k]->kernel().l = l;
            //     // pivx[k].compute ( * ( Kx[k] ), tol );
            //     // indexSet.insert(pivx[k].pivots().cbegin(), pivx[k].pivots().cend());
            // }
//          common set of indices, should not be too many
            // pivots.resize(indexSet.size());
            // std::copy(indexSet.cbegin(), indexSet.cend(), pivots.begin());
            alpha = Vector::Constant(m, 1.0/static_cast<double>(m));
            Matrix sumK = Matrix::Constant(n,n, 0.0);
            for ( unsigned int k = 0; k < m; ++k ) {
                sumK += alpha(k) * Kx[k];
            }
            // Matrix sumK = std::inner_product(alpha.cbegin(),alpha.cend() ,Kxblock.cbegin(),  Matrix::Constant(n,n, 0.0));
            sumK.diagonal().array() += static_cast<double>(n)*lambda; 
            
            
            c = sumK.llt().solve(y);
            
            double eps1(1);
            double eps2(2);
            unsigned int counter(1);
            Vector beta_1   = alpha;
            Vector gamma_1  = alpha;
            Vector delta_1  = Vector::Constant(m, 0);
            
            Vector beta_0;
            Vector gamma_0;
            Vector delta_0;
            
            Vector v(m);
            Matrix G(n,m);
            Matrix mulmat;
            Matrix sepp;
            const double nd(static_cast<double>(n));
            
            while(counter < 20000 && std::max(eps1, eps2)> 1.e-12){
                beta_0 = beta_1;
                gamma_0 = gamma_1;
                delta_0 = delta_1;
                for(unsigned int i = 0; i < m; ++i){
                    Vector aux = Kx[i].selfadjointView<Eigen::Lower>() * c;
                    v(i) = c.dot(aux);
                    G.col(i) = aux;
                }
                mulmat = G.transpose() * G;
                sepp = rho/nd * mulmat;
                sepp.diagonal() +=mulmat.diagonal() + Vector::Ones(m);
                beta_1  = sepp.llt().solve(2.0*rho/nd * G.transpose() * y-lambda*rho*v+gamma_0-delta_0);
                gamma_1 = simplex_project(beta_1+delta_0);
                delta_1 = delta_0 + beta_1-gamma_1;
                
                sumK = Matrix::Constant(n,n, 0.0);
                for ( unsigned int k = 0; k < m; ++k ) {
                    sumK += gamma_1(k) * Kx[k];
                }
                
                // sumK = std::inner_product(Kxblock.cbegin(), Kxblock.cend(), gamma_1.begin(), Matrix::Constant(n,pivots.size(), 0.0));
                sumK.diagonal().array() += static_cast<double>(n)*lambda; 
                c = sumK.llt().solve(y);
                
                
                
                eps1 = (beta_1-beta_0).cwiseAbs().sum();
                eps2 = (delta_1-delta_0).cwiseAbs().sum();
                std::cout << eps1 << '\t' << eps2 << std::endl;
                ++counter;
            }
            alpha = gamma_1;
            // std::cout << alpha << std::endl;
            return(EXIT_SUCCESS);
        }
       const Vector& getAlpha() const {return alpha;} 
       const Vector& getC() const {return c;} 
       double getObj() const {
           Matrix sumK = Matrix::Constant(n,n, 0.0);
            for ( unsigned int k = 0; k < m; ++k ) {
                sumK += alpha(k) * Kx[k];
            }
            return((y-sumK.selfadjointView<Eigen::Lower>() * c ).squaredNorm());
       }
        

    private:
        const Vector& y;
        const Matrix& X;
        
        
        const unsigned int n; // how many data points do we have
        const unsigned int m; // how large is the dimension of X
        
        const Matrix Xt; // the transpose of X
        
        
        Vector alpha; // Vector of MKL weights
        Vector c; //coefficient vector
        
        std::vector<Matrix> Kx;
        

    
    };
    
    
    
    
    
     
    
} // RRCA

#endif
