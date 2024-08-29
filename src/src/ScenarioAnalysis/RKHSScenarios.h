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
#ifndef RRCA_SCENARIOANALYSIS_RKHSSCENARIOS_H_
#define RRCA_SCENARIOANALYSIS_RKHSSCENARIOS_H_


namespace RRCA
{
namespace SCENARIOANALYSIS
{
   
    
    
    
    class RKHSScenarios {
       
    public:
    /*
    *    \brief Given moment matrix A and empirical scenarios pts computes least squares probabilities that fit to A 
   */
        RKHSScenarios(const Matrix& pts_, unsigned int order_, double tol_= 0.0) : 
        pts(pts_), 
        order(order_), 
        datadim(pts_.rows()), // assuming that data is in col major order (time series along the x-axis
        n(pts_.cols()),
        basisdim(binomialCoefficient ( 2*order_ + pts_.rows(), pts_.rows()  )),
        smallBasisdim(binomialCoefficient ( order_ + pts_.rows(), pts_.rows()  )),
        tol(tol_),
        myIndex ( pts_.rows(),2*order_ ),
        mySmallIndex ( pts_.rows(),order_ ),
        K(pts_),
        bas(K,piv),
        V(basisdim,n),
        resP(n) {
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
            K.kernel().init(pts_.rows(),2*order_);
            
//             std::cout << scen << std::endl;

        }
        int solve(){
            A = (V * V.transpose()/static_cast<double>(pts.cols()));
            K.kernel().setM(A.inverse());
            piv.compute ( K,tol );
            bas.init(K, piv.pivots());
            scen = bas.src_pts();
            const auto &mySet = mySmallIndex.get_MultiIndexSet();
            smallA = A.topLeftCorner(mySet.size(),mySet.size());
            
//             LeastSquaresProbabilities<Matrix> ls(scen, order,smallA);
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
        
        
        RRCA::KernelMatrix<RRCA::PolynomialKernel, Eigen::MatrixXd> K;
        RRCA::PivotedCholeskyDecompositon<RRCA::KernelMatrix<RRCA::PolynomialKernel, Eigen::MatrixXd> > piv;
        RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::PolynomialKernel, Eigen::MatrixXd> >  bas;
        
        Matrix A;
        Matrix smallA;
        Matrix V;
        Matrix scen;
        Vector resP;
    
    };
    
  
    
    
    
    
    
     
    
} // SCENARIOANALYSIS
} // RRCA

#endif
