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
#ifndef RRCA_SCENARIOANALYSIS_OMPSCENARIOS_H_
#define RRCA_SCENARIOANALYSIS_OMPSCENARIOS_H_

namespace RRCA
{
namespace SCENARIOANALYSIS
{
   
    
    
    
    class OMPScenarios {


    
    public:
    /*
    *    \brief Given moment matrix A and empirical scenarios pts computes least squares probabilities that fit to A 
    solving here for the vector-valued representation of the moment matrix that has moments of order 2*order. for this reason we need to multiply order by two. 
   */
        OMPScenarios(const Matrix& pts_, unsigned int order_, double tol_= 0.0) : 
        pts(pts_), 
        order(2*order_), 
        datadim(pts_.rows()), // assuming that data is in col major order (time series along the x-axis
        n(pts_.cols()),
        basisdim(binomialCoefficient ( 2*order_+ pts_.rows(), pts_.rows()  )),
        tol(tol_),
        myIndex ( pts_.rows(),2*order_ ),
        V_t(basisdim,n) {
            std::cout << " pts rows " << pts_.rows() << " prs cols " << pts_.cols() << std::endl;
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
                    V_t(counter, i) = accum;
                    ++counter;
                }
            }
            y = V_t.rowwise().mean();
        }
        const Matrix& getA() const {
            return(smallA);
        }
        int solve(){
            A = (V_t * V_t.transpose()/static_cast<double>(pts.cols()));
            const unsigned int SMALL(binomialCoefficient ( order/2+ pts.rows(), pts.rows()  ));
            smallA = A.topLeftCorner(SMALL,SMALL);
            RRCA::OMPCholeskyDecompositon<Matrix> chol(V_t.transpose()*V_t, V_t.transpose()* y, 0.0, MAXITER);
            scen = pts(Eigen::all, chol.pivots());
            scenV_t = V_t(Eigen::all, chol.pivots());
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
        
        const Matrix& getV_t() const {return(V_t);}
        const Matrix& getScenV_t() const {return(scenV_t);}
        const Vector& gety() const {return(y);}
        double getTol() const {return(tol);}
        
//         Matrix getScenA(const Vector& resP){
//             const auto &mySet = mySmallIndex.get_MultiIndexSet();
// //          copy into vector for simplicity
//             unsigned int counter;
//             Matrix scenV(smallBasisdim,scen.cols());
// //             fill V matrix
//             for(unsigned int i = 0; i < scen.cols(); ++i){
//                 counter = 0;
//                 double accum(1);
//                 for ( const auto &ind1 : mySet ) {
//                     accum = 1;
//                     for(unsigned int j = 0; j < datadim; ++j){
//                         accum *= pow(scen(j,i),ind1(j));
//                     }
//                     scenV(counter, i) = accum;
//                     ++counter;
//                 }
//             }
// 
//             return(scenV * resP.asDiagonal() * scenV.transpose());
//         }

        
    private:
        const Matrix &pts;
        const unsigned int order; // what is the order
        const unsigned int datadim; // how many variables
        const unsigned int n; // how many data points do we have
        const unsigned int basisdim; // how many monomials do we have in the monomial basis
        double tol;
        
        const RRCA::MultiIndexSet<iVector> myIndex;
        
        
        
        Matrix A;
        Matrix smallA;
        Matrix V_t;
        Matrix scenV_t;
        Matrix scen;
        Vector y;
//         Vector resP;
        
        const unsigned int MAXITER = 250;
    
    };
    
  
    
    
    
    
    
     
    
} // SCENARIOANALYSIS
} // RRCA

#endif
