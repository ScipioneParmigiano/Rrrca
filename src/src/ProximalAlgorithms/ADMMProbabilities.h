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
#ifndef RRCA_PROXIMALALGORITHMS_ADMMPROBABILITIES_H_
#define RRCA_PROXIMALALGORITHMS_ADMMPROBABILITIES_H_



namespace RRCA
{
namespace PROXIMALALGORITHMS
{
    
//      template<typename Derived> 
    class ADMMProbabilities {
    
    public:
    /*
    *    \brief Minimizes \|V_t_ x - y\|_2^2 s.t. x>=0 sum(x)=1
   */
        ADMMProbabilities(const Matrix& V_t_, const Vector &y_) :
        V_t(V_t_),
        yy(y_),
        n(V_t_.cols()),
        basisdim(V_t_.rows()),
        lambda(n) {
            assert(y_.size() == basisdim);
            // std::cout << "V_t:\n" << V_t_ << std::endl;
            // std::cout << "y_:\n" << y_ << std::endl;
        }
        const Vector& getProbs() const {
            return(lambda);
        }
        int solve(){
            lambda.array() = 1;
            const unsigned int iters = 200;

            const Vector A_m       = V_t.rowwise().mean();
            const Matrix AA            = V_t - A_m*RowVector::Ones(n);
            const Vector yaux         = yy - A_m;
            const double norm_y       = (yaux.norm()<=0.0 ? 1e-10 : yaux.norm());
            
            const Matrix A           = AA/norm_y;
            const Vector y            = yaux/norm_y;
            lambda                    /= norm_y*norm_y;
            
            const Vector a            = Vector::Ones(n);
            double mu = 10*lambda.mean() + 0.01;
            // Eigen::BDCSVD<Matrix> svd(V_t.transpose()*V_t,Eigen::ComputeThinU | Eigen::ComputeThinV);
            // const Vector vals       = svd.singularValues();
            
            Eigen::SelfAdjointEigenSolver<Matrix> es(A.transpose()*A);
            const Vector vals       = es.eigenvalues();
            const Matrix vecs       = es.eigenvectors();
            
            Vector valcomb    = (vals.array()+mu).matrix();
            Matrix inv_B      = vecs * valcomb.cwiseInverse().asDiagonal() * vecs.transpose();
            
            // std::cout << "inv_B:\n " << inv_B << std::endl;


            Vector C = (inv_B*a)/(a.transpose()*inv_B*a); 
            // std::cout << "C:\n:" << C << std::endl; 
            Matrix D = (inv_B - C*a.transpose()*inv_B);
            // std::cout << "D:\n:" << D << std::endl; 
            Vector b = A.transpose()*y;
            // std::cout << "b:\n" << b << std::endl;
            
//          now the algo 
            Vector u    = Vector::Zero(n);
            Vector u0;
            Vector d    = Vector::Zero(n);
            double tol_1 = sqrt(static_cast<double>(n))*1e-10;
            double tol_2 = sqrt(static_cast<double>(n))*1e-10;
            double res_p = INFINITY;
            double res_d = INFINITY;
            Vector x;

            unsigned int k = 1;
            unsigned int mu_changed = 0;
            
            while ((k <= iters) && ((abs(res_p) > tol_1) || (abs(res_d) > tol_2))){    
                if (k % 10 == 1){
                    u0 = u;
                }

                Vector w = b + mu*(u + d);
                // std::cout << "w:\n" << w << std::endl;
                x = D*w + C;
                // std::cout << "x:\n" << x << std::endl;
                // Vector inter = soft(x-d,lambda/mu);
                // Vector u_pos = (u >= 0.0);
                u = soft(x-d,lambda/mu).unaryExpr([](double xx) { return (xx>=0.0 ? xx : 0.0); }); 
                // std::cout << "u:\n" << u << std::endl;
                d -= (x-u);
                // std::cout << "d:\n" << d << std::endl;
                
                

                // update mu so to keep primal and dual residuals within a factor of 10
                if (k % 10 == 1){

                    res_p = (x-u).norm();
                    res_d = mu * (u-u0).norm();
                    // std::cout << "k: " << k << " res_p: " << res_p << " res_d" << res_d << std::endl;
                    // std::cout << "d:\n" << d << std::endl;
                    // std::cout << "mu: " << mu << std::endl;
        
                    if (res_p > 10.0*res_d){
                        mu *= 2.0;
                        d  /=2.0;
                        mu_changed = 1;
                    } else if (res_d > 10.0*res_p){
                        mu /= 2.0;
                        d  *=  2.0;
                        mu_changed = 1;
                    }
                    if  (mu_changed>0.0){
                        // std::cout << "mu changed d: " << d << std::endl;
                        valcomb    = vals.array()+mu;
                        inv_B      = vecs * valcomb.cwiseInverse().asDiagonal() * vecs.transpose();
                        C = (inv_B*a)/(a.transpose()*inv_B*a);              
                        D = (inv_B - C*a.transpose()*inv_B);
                        mu_changed = 0;
                    }
                }

                ++k;   
            }
            // if(x.minCoeff() < -1.0 || x.maxCoeff() > 1.5){ // did not converge
            //     return(EXIT_FAILURE);
            // } else if (x.minCoeff() < 0.0){
            //     lambda = x.unaryExpr([](double xx) { return (xx<0.0 ? 0.0 : xx); }); 
            // } else {
            //     lambda = x;
            // }
            lambda = u;
            return(EXIT_SUCCESS);
        }
        

    private:
        const Matrix& V_t;
        const Vector&  yy;
        
        const unsigned int n; // how many data points do we have
        const unsigned int basisdim; // how many monomials do we have in the monomial basis
        
        
        Vector lambda;
        
        Vector soft(const Vector& vec, const Vector& tau) const {
            return(vec.cwiseSign().cwiseProduct(vec.cwiseAbs()-tau));
            // Vector T = tau.array() + 1e-6;
            // Vector y = (vec.cwiseAbs()-T).unaryExpr([](double x) { return (x>=0.0 ? x : 0.0); });
            // return(((y.cwiseQuotient(y+T)).cwiseProduct(vec)));
        }
        

        
        

    
    };
    
    
    
    
    
     
    
} // PROXIMALALGORITHMS
} // RRCA

#endif
