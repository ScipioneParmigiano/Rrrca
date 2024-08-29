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
#ifndef RRCA_SCENARIOANALYSIS_MAXVOLSCENARIOS_H_
#define RRCA_SCENARIOANALYSIS_MAXVOLSCENARIOS_H_

// #include <eigen3/Eigen/Dense>
namespace RRCA
{
namespace SCENARIOANALYSIS
{
   
    
    
    
    class MaxVolScenarios {


    
    public:
    /*
    *    \brief Given moment matrix A and empirical scenarios pts computes least squares probabilities that fit to A 
   */
        MaxVolScenarios(const Matrix& pts_, unsigned int order_, double tol_= 0.0) : 
        pts(pts_), 
        order(order_), 
        datadim(pts_.rows()), // assuming that data is in col major order (time series along the x-axis
        n(pts_.cols()),
        basisdim(binomialCoefficient ( 2* order_+ pts_.rows(), pts_.rows()  )),
        smallBasisdim(binomialCoefficient ( order_ + pts_.rows(), pts_.rows()  )),
        tol(tol_),
        myIndex ( pts_.rows(),2*order_  ),
        mySmallIndex ( pts_.rows(),order_ ),
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
            }
            y = V.rowwise().mean();
        }
        int solve(){
            A = (V * V.transpose()/static_cast<double>(pts.cols()));
            A(0,0) = 1;
            Matrix VV = V.transpose();
            Matrix iden = Matrix::Identity(VV.rows(),VV.cols());
            
            for(unsigned int i = 0; i < 2; ++i){
                Eigen::HouseholderQR<RRCA::Matrix> qr(VV);
                // std::cout << qr.householderQ().rows() << '\t' << qr.householderQ().cols() << std::endl;
                VV = qr.householderQ() * iden;
                // std::cout << VV.rows() << '\t' << VV.cols() << std::endl;
            }
            
            Vector w = VV.transpose().colPivHouseholderQr().solve(Vector::Ones(V.rows()));
            
            std::vector<unsigned int> pivs;
//          find nonzero entries and write them into a vector
            for(unsigned i = 0; i < pts.cols(); ++i){
                if(w(i) > 0.0 || w(i) < 0.0){
                    pivs.push_back(i);
                }
            }
            
            scen = pts(Eigen::all, pivs);
            scenV = V(Eigen::all, pivs);
//             std::cout << scen << std::endl;
//             const auto &mySet = mySmallIndex.get_MultiIndexSet();
            smallA = A.topLeftCorner(smallBasisdim,smallBasisdim);
            
//             LeastSquaresProbabilities ls(scen, order,smallA);
//             int success = ls.solveForWeights();
//             resP = ls.getProbs();
            
            return(EXIT_SUCCESS);
        }
//         Vector getProbs() const {
//             return(resP);
//         }
        Matrix getScenarios() const {
            return(scen);
        }
        void setTol(double tol_){
            tol = tol_;
        }
        double getTol() const {return(tol);}
        
        const Matrix& getA() const {return(smallA);}
        const Vector& gety() const{return(y);}
        
        const Matrix& getV_t() const {return(V);}
        const Matrix& getScenV_t() const {return(scenV);}
        
        Matrix getScenA(const Vector& resP){
            const auto &mySet = mySmallIndex.get_MultiIndexSet();
//          copy into vector for simplicity
            unsigned int counter;
            Matrix scenV(smallBasisdim,scen.cols());
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

            return(scenV * resP.asDiagonal() * scenV.transpose());
        }

        
    private:
        const Matrix &pts;
        const unsigned int order; // what is the order
        const unsigned int datadim; // how many variables
        const unsigned int n; // how many data points do we have
        const unsigned int basisdim; // how many monomials do we have in the monomial basis
        const unsigned int smallBasisdim; // how many monomials do we have in the monomial basis
        double tol;
        
        const RRCA::MultiIndexSet<iVector> myIndex;
        const RRCA::MultiIndexSet<iVector> mySmallIndex;
        
        
        
        Matrix A;
        Matrix smallA;
        Matrix V;
        Matrix scenV;
        Matrix scen;
        Vector y;
//         Vector resP;
    
    };
    
  
    
    
    
    
    
     
    
} // SCENARIOANALYSIS
} // RRCA

#endif
