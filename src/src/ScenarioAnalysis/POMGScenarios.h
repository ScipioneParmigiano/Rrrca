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
#ifndef RRCA_SCENARIOANALYSIS_PROXIMALOPTIMALGRADIENT_H_
#define RRCA_SCENARIOANALYSIS_PROXIMALOPTIMALGRADIENT_H_



namespace RRCA {
namespace SCENARIOANALYSIS {

class POMGScenarios {



public:
    /*
    *    \brief Given moment matrix A and empirical scenarios pts computes least squares probabilities that fit to A
    solving here for the vector-valued representation of the moment matrix that has moments of order 2*order. for this reason we need to multiply order by two.
    */
    POMGScenarios ( const Matrix& pts_, unsigned int order_, double tol_= 1e-4 ) :
        pts ( pts_ ),
        order ( 2*order_ ),
        datadim ( pts_.rows() ), // assuming that data is in col major order (time series along the x-axis
        n ( pts_.cols() ),
        basisdim ( binomialCoefficient ( 2*order_+ pts_.rows(), pts_.rows() ) ),
        tol ( tol_ ),
        myIndex ( pts_.rows(),2*order_ ),
        V_t ( basisdim,n ) {
//          compute the Vandermonde matrix for 2q
        const auto &mySet = myIndex.get_MultiIndexSet();
//          copy into vector for simplicity
        unsigned int counter;
//             fill V matrix
        for ( unsigned int i = 0; i < n; ++i ) {
            counter = 0;
            double accum ( 1 );
            for ( const auto &ind1 : mySet ) {
                accum = 1;
                for ( unsigned int j = 0; j < datadim; ++j ) {
                    accum *= pow ( pts ( j,i ),ind1 ( j ) );
                }
                V_t ( counter, i ) = accum;
                ++counter;
            }
        }
        y = V_t.rowwise().mean();

//          now compute Q and y tilde
        A = ( V_t * V_t.transpose() /static_cast<double> ( pts.cols() ) );
        const unsigned int SMALL ( binomialCoefficient ( order/2+ pts.rows(), pts.rows() ) );
        smallA = A.topLeftCorner ( SMALL,SMALL );

        RRCA::PivotedCholeskyDecompositon<Matrix> piv ( A,0.0 );
        piv.computeBiorthogonalBasis();
        B = piv.matrixB();
        Minv    = B * B.transpose();
        y_til   = B.transpose() * y/sqrt ( static_cast<double> ( n ) );
        Q = V_t.transpose() * B/sqrt ( static_cast<double> ( n ) );
        
        
    }
    const Matrix& getA() const {
        return ( smallA );
    }
    int solve() {
//          now run the POMG algo and the ADMM until the Lcurve error does not decrease sufficiently anymore
        const unsigned int partitions ( 30 );
        unsigned int counter = 1;
        double paretodecrease = 1;
        const unsigned int SMALL ( binomialCoefficient ( order/2+ pts.rows(), pts.rows() ) );
        Vector loglambda = Vector::LinSpaced ( 30,log ( 1.e-8 ), log ( 2.e-4 ) );
        loglambda.reverseInPlace();


        const Matrix K = Q * Q.transpose();
        const Vector b = Q * y_til;

        unsigned int i = 0; 
        
        // Eigen::SelfAdjointEigenSolver<Matrix> eig(Q*Q.transpose(), Eigen::EigenvaluesOnly);
        // const double L    = eig.eigenvalues().maxCoeff();
        const double L = sqrt((Q*Q.transpose()).diagonal().sum());
        
        while ( i < loglambda.size() &&  paretodecrease > tol ) {
            double lambda = exp(loglambda ( i ));
            
            double t = 1;
            double t0 =1;

            Vector x = Vector::Zero ( n );
            Vector w = Vector::Zero ( n );
            Vector z = Vector::Zero ( n );
            
            // Vector x0;
            Vector w0 = w;
            // Vector z0;


            double F    = 0.5 * y.squaredNorm();
            double F0   = F;
            Vector G    = - b;
            Vector G0;
            Vector yy; 
            Vector x0; 
            
            double obj;
            double obj0(1000000);

            double sig  = 1;
            double bsig = 0.8;
            double zeta = 1;
            double gamma;
            double beta;
            unsigned int counter = 0;

            while ( counter <= 250 && abs(obj-obj0) > tol  ) {

                std::cout << " counter " << counter << " F " << F << std::endl;
                t0 = t;
                F0 = F;
                G0 = G;
                x0 = x;
                w0 = w;
                obj0 = obj;
                

                
                

//                 gradient step
                w       = x0- ( 1.0/L ) * ( K*x0-b );
                
                if ( counter < MAXITER ) {
                    t = 0.5 * ( 1.0+sqrt ( 4.0*t0*t0+1.0 ) );
                } else {
                    t = 0.5 * ( 1.0+sqrt ( 8.0*t0*t0+1.0 ) );
                }
                
                beta    = ( t0-1.0 ) /t;
                gamma   = ( sig*t0 ) /t;

//                 momentum step
                z       = w + beta * ( w-w0 )+gamma* ( w-x0 )-  beta /(L*zeta)  * ( x0 - z );

                zeta    = ( 1.0+beta+gamma ) /L;
                
                
                

                
//                 proximity step
                x       = softthresholding ( z, Vector::Constant(z.size(),zeta*lambda) );

//                 composite error function
                F = 0.5 * (Q.transpose() * x - y_til).squaredNorm()+lambda * x.cwiseAbs().sum();
                
                std::cout << " F " << F <<  " F0 " << F0 << std::endl; 
                
                //                 composite gradient mapping
                G = ( K*x0-b )-1.0/zeta* ( x-z );
                yy = x0 -1.0/L * G;
                std::cout << "y" << yy.head(5).transpose() << std::endl;
                obj = 0.5 * (Q.transpose() * yy - y_til).squaredNorm()+lambda * yy.cwiseAbs().sum(); 
                std::cout << "objective function new " << obj << std::endl;
                
                

//                 restart condition
                if ( F > F0 ) {
                    t = 1;
                    sig = 1;
                } else  if ( ( G.cwiseProduct ( G0 ) ).sum() < 0.0 ) { //                 decreasing gamma condition
                    sig = bsig*sig;
                }
                

                
                ++counter;
            } // end POMG algo

//             now run admm on the resulting scenarios
            std::vector<unsigned int> inds;
//          find nonzero entries and write them into a vector
            for(unsigned i = 0; i < y.size(); ++i){
                if(y(i) > 0.0 || y(i) < 0.0){
                    inds.push_back(i);
                }
            }



//              run ADMM every 10 iterations to check the criterion
            scenV_t = V_t ( Eigen::all, inds );
            scen = pts ( Eigen::all,inds );
            RRCA::PROXIMALALGORITHMS::ADMMProbabilities admm ( scenV_t,y );
            admm.solve();
            theProbs = admm.getProbs();
            Matrix model = scenV_t.topRows ( SMALL ) *  admm.getProbs().asDiagonal() *  scenV_t.topRows ( SMALL ).transpose();
            double relErr = (smallA-model).norm()/smallA.norm();
            paretodecrease = relErr/(static_cast<double>(inds.size()));
            std::cout << "relerr " << relErr << '\t' << "size " << inds.size() << std::endl;
            ++i;
        }
        return ( EXIT_SUCCESS );
    }
    const Vector& getProbs() const {
        return ( theProbs );
    }
    Matrix getScenarios() const {
        return ( scen );
    }
    void setTol ( double tol_ ) {
        tol = tol_;
    }

    Matrix getScenA() const {
        const unsigned int SMALL(binomialCoefficient ( order/2+ pts.rows(), pts.rows()  ));
            return((scenV_t * theProbs.asDiagonal() * scenV_t.transpose()).topLeftCorner(SMALL,SMALL));
    }

    const Matrix& getV_t() const {
        return ( V_t );
    }
    const Matrix& getScenV_t() const {
        return ( scenV_t );
    }
    const Vector& gety() const {
        return ( y );
    }
    double getTol() const {
        return ( tol );
    }



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


    Vector softthresholding ( const Vector& vec, const Vector& tau ) const {
        // return ( vec.cwiseSign().cwiseProduct ( vec.cwiseAbs()-tau ) );
        return ( vec.cwiseSign().cwiseProduct ( (vec.cwiseAbs()-tau).cwiseMax(0.0) ) );
    }

    double err ( const Vector& x, double lambda ) const {
        return ( 0.5* ( Q.transpose() * x-y_til ).squaredNorm()+lambda*x.cwiseAbs().sum() );
    }

};









} // SCENARIOANALYSIS
} // RRCA

#endif
