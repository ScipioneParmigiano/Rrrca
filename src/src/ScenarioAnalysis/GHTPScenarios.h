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
#ifndef RRCA_SCENARIOANALYSIS_GRADEDHARDTHRESHOLDINGPURSUIT_H_
#define RRCA_SCENARIOANALYSIS_GRADEDHARDTHRESHOLDINGPURSUIT_H_



namespace RRCA
{
namespace SCENARIOANALYSIS
{
    
    class GHTPScenarios {


    
    public:
    /*
    *    \brief Given moment matrix A and empirical scenarios pts computes least squares probabilities that fit to A 
    solving here for the vector-valued representation of the moment matrix that has moments of order 2*order. for this reason we need to multiply order by two. 
   */
        GHTPScenarios(const Matrix& pts_, unsigned int order_, double tol_= 1e-8) : 
        pts(pts_), 
        order(2*order_), 
        datadim(pts_.rows()), // assuming that data is in col major order (time series along the x-axis
        n(pts_.cols()),
        basisdim(binomialCoefficient ( 2*order_+ pts_.rows(), pts_.rows()  )),
        tol(tol_),
        myIndex ( pts_.rows(),2*order_ ),
        V_t(basisdim,n) {
//          compute the Vandermonde matrix for 2q
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
            
//          now compute Q and y tilde
            A = (V_t * V_t.transpose()/static_cast<double>(pts.cols()));
            const unsigned int SMALL(binomialCoefficient ( order/2+ pts.rows(), pts.rows()  ));
            smallA = A.topLeftCorner(SMALL,SMALL);
            
            RRCA::PivotedCholeskyDecompositon<Matrix> piv(A,0.0);
            piv.computeBiorthogonalBasis();
            B = piv.matrixB();
            Minv    = B * B.transpose();
            y_til   = B.transpose() * y/sqrt(static_cast<double>(n));
            Q = V_t.transpose() * B/sqrt(static_cast<double>(n));
        }
        const Matrix& getA() const {
            return(smallA);
        }
        int solve(){
//          now run the GHTP algo and the ADMM until the Lcurve error does not decrease sufficiently anymore
            unsigned int counter = 0;
            double paretodecrease = 100000;
            const unsigned int SMALL(binomialCoefficient ( order/2+ pts.rows(), pts.rows()  ));
            
            Vector x = Vector::Zero(n);
            Vector v = Vector::Zero(n);
            
            iVector S;
            iVector idx;
            
            while (counter <= MAXITER &&  paretodecrease > tol){
                // std::cout << "counter " << counter << "paretodecrease " << paretodecrease << std::endl;
                
                v = x + Q * (y_til - Q.transpose() * x);
                idx = iVector::LinSpaced(n, 0, n-1);
                std::stable_sort(idx.begin(), idx.end(), [&v](unsigned int i1, unsigned int i2) {return v[i1] < v[i2];});
                S = idx.tail(counter+1);
                x = Vector::Zero(n);
                
                v = (Q(S,Eigen::all) * Q(S,Eigen::all).transpose()).llt().solve(Q(S,Eigen::all) * y_til);
                std::vector<unsigned int> inds;
//              find nonzero entries and write them into a vector
                for(unsigned i = 0; i < v.size(); ++i){
                    if(v(i) > 0.0 || v(i) < 0.0){
                        inds.push_back(S(i));
                        x(S(i)) = v(i);
                    }
                }
                
//              run ADMM every 10 iterations to check the criterion
                if(counter % 5 == 0){
                    // std::cout << " in the ADMM step with indices " << inds.size() << std::endl; 
                    scenV_t = V_t(Eigen::all, inds);
                    scen = pts(Eigen::all,inds);
                    RRCA::PROXIMALALGORITHMS::ADMMProbabilities admm(scenV_t,y);
                    admm.solve();
                    theProbs = admm.getProbs();
                    Matrix model = scenV_t.topRows(SMALL) *  admm.getProbs().asDiagonal() *  scenV_t.topRows(SMALL).transpose();
                    double relErr = (smallA-model).norm()/smallA.norm();
                    paretodecrease = relErr/(static_cast<double>(inds.size()));
                }
                counter+=5;
            }
            // std::cout << "counter" << '\t' << counter << std::endl;
            return(EXIT_SUCCESS);
        }
        const Vector& getProbs() const {
            return(theProbs);
        }
        Matrix getScenarios() const {
            return(scen);
        }
        void setTol(double tol_){
            tol = tol_;
        }
        
        Matrix getScenA() const {
            const unsigned int SMALL(binomialCoefficient ( order/2+ pts.rows(), pts.rows()  ));
            return((scenV_t * theProbs.asDiagonal() * scenV_t.transpose()).topLeftCorner(SMALL,SMALL));
        }
        
        const Matrix& getV_t() const {return(V_t);}
        const Matrix& getScenV_t() const {return(scenV_t);}
        const Vector& gety() const {return(y);}
        double getTol() const {return(tol);}
        

        
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

        Matrix Q;
        Vector y_til;
        Matrix B;
        Matrix Minv;
        
        Vector theProbs;
        
        const unsigned int MAXITER = 250;
    
    };
    
  
    
    
    
    
    
     
    
} // SCENARIOANALYSIS
} // RRCA

#endif
