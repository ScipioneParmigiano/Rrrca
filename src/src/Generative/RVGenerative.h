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
#ifndef RRCA_GENERATIVE_PICTURE_GENERATIVE_H_
#define RRCA_GENERATIVE_PICTURE_GENERATIVE_H_



namespace RRCA {
namespace GENERATIVE {

//     draws from the distribution of assets. input is a balanced panel. uses low-rank framework
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class RVGenerative {
// data is colwise
public:
    RVGenerative ( const Matrix& data_, unsigned int sims = 0, unsigned int seed  =   time ( 0 ) ) :
        data ( data_ ),
        n ( data_.cols() ),
        nd ( data_.cols() ),
        d ( data_.rows() ),
        l ( ( sims ==0 ? n : sims ) ),
        ld ( l ),
        simData ( d,l ),
        K ( simData ),
        bas ( K,piv ),
        mu ( data_.rowwise().mean() ),
        sigma ( data_ * data_.transpose() /nd - mu * mu.transpose() ) {
        Eigen::SelfAdjointEigenSolver<Matrix> eig ( sigma );
        sigmainv = eig.eigenvectors() * eig.eigenvalues().cwiseInverse().asDiagonal() * eig.eigenvectors().transpose();
        detsigma = eig.eigenvalues().cwiseSqrt().prod();

        std::normal_distribution<> dis;
        std::default_random_engine gen ( seed );
        auto uni = [&]() {
            return dis ( gen );
        };
//
        Matrix inter = Matrix::NullaryExpr ( d, l,uni );
        simData =  eig.operatorSqrt() * rowStandardize ( inter );
        // std::cout << " simData mean before scaling " << simData.rowwise().mean() << std::endl;
        simData.colwise() += mu;
        // std::cout << simData << std::endl;
//          check distributon of simdata
        Vector musim = simData.rowwise().mean();
        Matrix simSigma = simData * simData.transpose() /ld - musim * musim.transpose();
        // std::cout << "mu : " << std::endl << mu << std::endl;
        // std::cout << "musim : " << std::endl << musim << std::endl;

        // std::cout << "sigma : " << std::endl << sigma << std::endl;
        // std::cout << "sigmasim : " << std::endl << simSigma << std::endl;

        fac = exp ( -0.5 * mu.transpose() * sigmainv * mu );

    }



    unsigned int getSubspaceDimension() const {
        return ( L.cols() );
    }

    Vector evaluateInSample ( double l, double prec,double lam ) {
        solveUnconstrained ( l,prec,lam );
        KernelMatrix K2 ( data );
        K2.kernel().l = l;
        KernelBasis bas2;
        Vector part1 = bas2.evalfull ( K2,simData ).colwise().mean().transpose() /lam;
        Vector part2 = bas.eval ( simData ) * u /ld;

        std::cout << " mean " << ( part1+part2 ).mean() << std::endl;
        return ( part1+part2 );
    }





    /*
    *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
    */
    int solveUnconstrained ( double l, double prec,double lam ) {

        precomputeKernelMatrices ( l, prec );
        u = - Q * ( L.transpose() * Kcal_t_1 ).cwiseQuotient ( bas.vectorLambda() /ld+Vector::Constant ( L.cols(), lam ) ) / ( lam*nd );

        // std::cout << " u " << u << std::endl;
        // std::cout << " sum u " << u.sum() << std::endl;
        return ( EXIT_SUCCESS );
    }

    /*
    *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
    */
    int solve ( double l, double prec,double lam ) {

        return ( solveUnconstrained ( l,prec,lam ) );
    }
    /*
    *    \brief generate n samples from the mixture distributon
    */
    Matrix sample ( unsigned int numSamples,double l, double prec,double lam ) {
//         const unsigned int m(L.cols());
//         Eigen::SelfAdjointEigenSolver<Matrix> eig(sigmainv + 1.0/(l*l) * Matrix::Identity(d, d));
//
//         sigmatilde = eig.eigenvectors() * eig.eigenvalues().cwiseInverse().asDiagonal() * eig.eigenvectors().transpose() ;
//         double const multiplier = eig.eigenvalues().cwiseInverse().cwiseSqrt().prod() / (detsigma*nd);
//         // Matrix b = (data(Eigen::all,piv.pivots())/(l*l)+constvec.replicate(1,m));
//         Vector extras = (b.transpose()*sigmatilde*b).diagonal();
//
//         // std::cout << "eig.eigenvalues() " << eig.eigenvalues() << std::endl;
//         std::cout << "muliplier " << multiplier << std::endl;
//         Vector intheexponent = -0.5*((data(Eigen::all,piv.pivots()).transpose()*data(Eigen::all,piv.pivots())).diagonal()/(l*l)+Vector::Constant(m,fac)-extras);
//         v = u.cwiseProduct(intheexponent.array().exp().matrix())*multiplier;
//
//         std::cout << " v " << v << std::endl;
//         std::cout << " sum of v " << v.cwiseMax(0.0).sum() << '\t' << v.cwiseMin(0.0).sum() << std::endl;

        return ( v );
    }

    Matrix independenceMH ( unsigned int N, double l, double prec,double lam, unsigned int seed = time ( 0 ) ) {

        solveUnconstrained ( l,prec,lam );


        std::normal_distribution<> dis;
        std::uniform_real_distribution<> uni;
        std::default_random_engine gen ( seed );
        
        auto gauss = [&]() {
            return dis ( gen );
        };
        auto uniform = [&]() {
            return uni ( gen );
        };
//
        Matrix inter = Matrix::NullaryExpr ( d, N+1,gauss );
        Vector epsilon = Vector::NullaryExpr ( N,uniform );
        Vector accept(N);
        Vector xi_i_m1 = inter.col ( N );
        Eigen::SelfAdjointEigenSolver<Matrix> eig ( sigma );
        Matrix proposalData =  eig.operatorSqrt() * rowStandardize ( inter );
        proposalData.colwise() += mu;

        Matrix out ( d,N );


        KernelMatrix K2 ( data );
        K2.kernel().l = l;
        KernelBasis bas2;
        double part1;
        double part2;
        double g_old;
        double g_new;
        
        part1 = bas2.evalfull ( K2,xi_i_m1 ).mean() /lam;
        part2 = (bas.eval ( xi_i_m1 ) * u).sum() /ld;
        g_old = part1 + part2;

        for ( auto i = 0; i < N; ++i ) {
//                 compute g using the previos
            part1 = bas2.evalfull ( K2,proposalData.col(i) ).mean()/lam;
            part2 = (bas.eval ( proposalData.col(i)  ) * u).sum() /ld;
            g_new = part1 + part2;
            std::cout << i << '\t' << g_new/g_old << std::endl;
            if(epsilon(i) <= std::min(std::max(g_new,0.0)/std::max(g_old,0.0),1.0)){ //accept
                g_old = g_new;
                xi_i_m1 = proposalData.col(i);
                out.col(i) = proposalData.col(i);
            
                accept(i) = 1;
            } else {
                out.col(i) = xi_i_m1;
                accept(i) = 0;
            }
// //             always accept
//             out.col(i) = proposalData.col(i);
//             accept(i) = 1;
            
        }
        
        Matrix reallyout(N,d+1);
        reallyout << out.transpose(), accept;
        return(reallyout);
    }







private:
    const Matrix& data;
    const unsigned int n;
    double const nd;
    const unsigned int d;
    const unsigned int l;
    double const ld;

    Matrix simData;


    KernelMatrix K;
    LowRank piv;
    KernelBasis bas;

    Matrix Q;
    Matrix L;
    Vector Kcal_t_1;
    const Vector mu;
    const Matrix sigma;
    Matrix sigmainv;

    double fac;

    Vector mutilde;
    Matrix sigmatilde;;


    double detsigma;
    double detsigmatilde;
    double multiplier;
    // const Vector constvec;

    Vector u;
    Vector v;





    double tol;





//     these are the empirical bounds for the functions

//     /*
//    *    \brief computes the kernel basis and tensorizes it
//    */
    void precomputeKernelMatrices ( double l, double prec ) {
        K.kernel().l = l;

        piv.compute ( K, prec );
        bas.init ( K, piv.pivots() );
        bas.initSpectralBasisWeights ( piv );
        Q = bas.matrixQ();
        L = bas.eval ( simData ) * Q;

        // std::cout << "L" << L << std::endl;

        Kcal_t_1 = bas.evalfull ( K,data ).colwise().sum().transpose(); // this is an l vector

//      testing if \Lambda is correct
        std::cout << "vector lambda " << ( ( L.transpose() * L ).diagonal() - bas.vectorLambda() ).norm() << std::endl;


//      for sampling
        Eigen::SelfAdjointEigenSolver<Matrix> eig ( sigmainv + 1.0/ ( l*l ) * Matrix::Identity ( d, d ) );
        sigmatilde = eig.eigenvectors() * eig.eigenvalues().cwiseInverse().asDiagonal() * eig.eigenvectors().transpose() ;
        detsigmatilde = eig.eigenvalues().cwiseInverse().cwiseSqrt().prod();
        multiplier = fac * detsigmatilde / detsigma;
        // std::cout << "eig.eigenvalues() " << eig.eigenvalues() << std::endl;
        std::cout << "muliplier " << multiplier << std::endl;
    }
};

















} // namespace GENERATIVE
}  // namespace RRCA
#endif
