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



namespace RRCA
{
namespace GENERATIVE
{
    
//     draws from the distribution of assets. input is a balanced panel. 
template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class RVGenerative
{

public:
    PictureGenerative ( const Matrix& xdata_,const Matrix& ydata_) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        n((!static_cast<bool>(xdata_.cols() % 2) ? xdata_.cols()/2 : (xdata_.cols()-1)/2 )),
        zdata (xdata_.rows()+ydata_.rows(),n),
        zeta(xdata_.rows()+ydata_.rows(),n),
        Kzeta ( zeta ),
        baszeta(Kzeta,pivzeta),
        Kz ( zdata ),
        basz(Kz,pivz)
        {
            
            zdata.topRows(xdata_.rows())        = xdata_.leftCols(n);
            zdata.bottomRows(ydata_.rows())     = ydata_.leftCols(n);
            
            zeta.topRows(xdata_.rows())     = xdata_(Eigen::all, Eigen::seq(0, (!static_cast<bool>(xdata_.cols() % 2) ? xdata_.cols()-1 : xdata_.cols()-2 ) , 2));
            zeta.bottomRows(ydata_.rows())  = ydata_(Eigen::all, Eigen::seq(1, Eigen::last, 2) );
            
            iVector indices = iVector::LinSpaced(xdata_.cols(), 0, xdata_.cols() -1);

            iid_int = indices(Eigen::seqN(0,n,2) );
            yint = ydata_(Eigen::all, iid_int);


            
    }
    


    unsigned int getSubspaceDimension() const {
        return(Lzeta.cols());
    }
    

    
    
    
          /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
        int solveUnconstrained ( double l, double prec,double lam )
    {
        
        precomputeKernelMatrices ( l, prec,lam );
        const double nn(Lzeta.rows());
        toInvert = Lzeta.transpose()*Lzeta/nn;
        toInvert.diagonal().array() += lam;
        u_til_lambda = Bzeta * toInvert.llt().solve(Lzeta.transpose() *(-q_til_lambda-Vector::Constant(Lzeta.rows(),p)) )/nn;

        return ( EXIT_SUCCESS );
    }
    
              /*
   *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
   */
        int solve ( double l, double prec,double lam )
    {
        
        return(solveUnconstrained(l,prec,lam));
    }
    
        /*
   *    \brief returns the y values used for integration
   */
    const Matrix& get_y_int() const {
        return(yint);
    }
//     /*
//    *    \brief computes conditional expectation in one dimension in Y
//    */
//     Matrix condExpfY_X ( const std::vector<std::function<double ( double ) > >& funs, const Matrix& Xs ) const
//     {
//        
//                 
//          
//         return ( res );
//     }
//     
        /*
   *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
   */
    Matrix condExpfVec ( const Matrix& Xs, bool structuralYes = false ) const    {
//                 compute a block matrix where the Xs are on the column dimension and the Ys are on the row dim 
        Matrix hblock = Matrix::Constant(yint.cols(), Xs.cols(),p);
        Matrix arg(yint.rows() + Xs.rows(), Xs.cols());
        Matrix zetas;
        Matrix zs;
        
//      
        // for(auto j = 0; j < Xs.cols(); ++j){
        //     for(auto i = 0; i < yint.cols(); ++i){
        //         arg << Xs.col(j),yint.col(i);
        //         zetas = baszeta.eval(arg);
        //         zs = basz.eval(arg);
        //         hblock(i,j) += zetas.dot(u_til_lambda) + zs.dot(lambda_vec);
        //     }
        // }
        // construct kernel matrices first
        // for(auto j = 0; j < Xs.cols(); ++j){
            for(auto i = 0; i < yint.cols(); ++i){
                arg << Xs ,yint.col(i).replicate(1,Xs.cols());
                zetas = baszeta.eval(arg);
                zs = basz.eval(arg);
                hblock.row(i) += (zetas * u_til_lambda +  zs * lambda_vec).transpose();
            }
        // }
        
        // std::cout << "hblock " << std::endl  << hblock << std::endl;
        if(structuralYes){
            hblock.array().rowwise() /= hblock.cwiseMax(0.0).colwise().sum().array();
            return(hblock.cwiseMax(0.0));
        }
        return(hblock/static_cast<double>((yint.cols())));
        
    }
    
                /*
    *    \brief computes conditional expectation of all the rows in Ys
    */
    const Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs, bool structuralYes = false ) const {
        Matrix hblock = Matrix::Constant(yint.cols(), Xs.cols(),p);
        Matrix arg(yint.rows() + Xs.rows(), Xs.cols());
        Matrix zetas;
        Matrix zs;
        // Vector arg(zeta.rows());
        // RowVector zetas;
        // RowVector zs;
        // for(auto j = 0; j < Xs.cols(); ++j){
        //     for(auto i = 0; i < yint.cols(); ++i){
        //         arg << Xs.col(j),yint.col(i);
        //         zetas = baszeta.eval(arg);
        //         zs = basz.eval(arg);
        //         hblock(i,j) += zetas.dot(u_til_lambda) + zs.dot(lambda_vec);
        //     }
        // }
        for(auto i = 0; i < yint.cols(); ++i){
                arg << Xs ,yint.col(i).replicate(1,Xs.cols());
                zetas = baszeta.eval(arg);
                zs = basz.eval(arg);
                hblock.row(i) += (zetas * u_til_lambda +  zs * lambda_vec).transpose();
            }
        if(structuralYes){
            hblock.array().rowwise() /= hblock.cwiseMax(0.0).colwise().sum().array();
            return(Ys * hblock.cwiseMax(0.0));
        }
        
        return(Ys * hblock/static_cast<double>((yint.cols())));
    }
                        /*
    *    \brief returns the "realized objective function for the validatino sample"
    */
    double getScore(const Matrix& valx, const Matrix& valy, bool structuralYes){
        assert(valx.cols() == valy.cols());
        const unsigned int valn((!static_cast<bool>(valx.cols() % 2) ? valx.cols()/2 : (valx.cols()-1)/2 ));
        Matrix zval(valx.rows() + valy.rows(), valn);
        Matrix zetaval(valx.rows() + valy.rows(), valn);
        zval.topRows(valx.rows())       = valx.leftCols(valn);
        zval.bottomRows(valy.rows())    = valy.leftCols(valn);
        
            
        zetaval.topRows(valx.rows())     = valx(Eigen::all, Eigen::seq(0, (!static_cast<bool>(valx.cols() % 2) ? valx.cols()-1 : valx.cols()-2 ) , 2));
        zetaval.bottomRows(valy.rows())  = valy(Eigen::all, Eigen::seq(1, Eigen::last, 2) );
        
//         now we evaluate the objective function -2<J^*_{(X,Y)}1-J^*p_*,h>_H+<J^*Jh,h>_H at the sample points that are the function arguments
        Matrix z_zs = basz.eval(zval);
        Matrix z_zetas = baszeta.eval(zval);
        
        Matrix zeta_zs = basz.eval(zetaval);
        Matrix zeta_zetas = baszeta.eval(zetaval);
        
        Vector zvec    = (z_zetas * u_til_lambda +  z_zs * lambda_vec); // with respect to joint measure
        Vector zetavec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec); // with respect to independence measure
        
        //      now that we have the sample, we can evaluate the objective function (that was solved before at these points)
        if(structuralYes){
            return(-2.0*zvec.cwiseMax(-p).mean()+2.0*p*zetavec.cwiseMax(-p).mean()+(zetavec.cwiseMax(-p).cwiseProduct(zetavec.cwiseMax(-p))).mean());
        }
        
        return(-2.0*zvec.mean()+2.0*p*zetavec.mean()+(zetavec.cwiseProduct(zetavec)).mean());
    }
    
                            /*
    *    \brief returns the negative log likelihood - log 1/valn sum p(y_i,x_i) for the validation sample
    */
    double negLogLikeli(const Matrix& valx, const Matrix& valy, bool structuralYes){
        assert(valx.cols() == valy.cols());
//      compute first h(valx, valy)
        Matrix evalz(valx.rows() + valy.rows(), valy.cols());
        evalz.topRows(valx.rows()) = valx; 
        evalz.bottomRows(valy.rows()) = valy;
        
        Vector evalg;
        
        if(structuralYes){
            evalg = (baszeta.eval(evalz) * u_til_lambda +  basz.eval(evalz) * lambda_vec+Vector::Constant(valx.cols(), p)).cwiseMax(RRCA_ZERO_TOLERANCE);
        } else {
            evalg = baszeta.eval(evalz) * u_til_lambda +  basz.eval(evalz) * lambda_vec+Vector::Constant(valx.cols(), p);
        }
        
        
//         now compute the p(x_i) distribution
        Matrix hblock = Matrix::Constant(yint.cols(), valx.cols(),p);
        Matrix arg(yint.rows() + valx.rows(), valx.cols());
        Matrix zetas;
        Matrix zs;

        for(auto i = 0; i < yint.cols(); ++i){
                arg << valx ,yint.col(i).replicate(1,valx.cols());
                zetas = baszeta.eval(arg);
                zs = basz.eval(arg);
                hblock.row(i) += (zetas * u_til_lambda +  zs * lambda_vec).transpose();
            }
        if(structuralYes){
            return(-(evalg.cwiseQuotient(hblock.cwiseMax(RRCA_ZERO_TOLERANCE).colwise().mean().transpose())).array().log().mean());
        }
        
        return(-(evalg.cwiseQuotient(hblock.colwise().mean().transpose())).array().log().mean());
    }
    
                    /*
    *    \brief returns the test statistic as a function of the L2_prior norm
    */
//     double getpeta(double lam, bool structuralYes){
// //         double const C1(1.0);
// //         double const C2(1.0);
// //         double const C3(1.0);
// //         
// //         double const kinf(1.0); // sup of kernel function
// //         double const piinf(1.0); // sup of p^*
// //         double const C4(sqrt(C2*C3)+C1*(kinf+C3*piinf)*C3*sqrt(C2)*sqrt(kinf));
// //         double const C5(2.0*sqrt(2.0*log(2.0/eta)*kinf)*(1.0+piinf+s*sqrt(kinf)));
// //         double const C6(sqrt(kinf)*(C5+C4));
// //         double const C7(pow(2.0,-0.5)*s+C6);
//         
//         Matrix zeta_zs = basz.eval(zeta);
//         Matrix zeta_zetas = baszeta.eval(zeta);
//         
//         Vector zetavec;
//         if(structuralYes){
//             zetavec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec).cwiseMax(-p);
//         } else {
//             zetavec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec);
//         }
//         
//         
//     }
    
//                       /*
//     *    \brief Lower partial moments of order k=1,2,3,....
//     */
//     double LPMS(bool structuralYes, int k, const Matrix& test){
//         Vector LPM(n);
//         LPM.setZero();
//         
//         Matrix zeta_zs = basz.eval(zeta);
//         Matrix zeta_zetas = baszeta.eval(zeta);
//         const double nd(n);
//         const unsigned int d(zeta.rows());
//         
//         Vector hvec;
//         if(structuralYes){
//             hvec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec).cwiseMax(-p);
//         } else {
//             hvec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec);
//         }
//         hvec.array() += p;
//         double const oneoverk(1.0/static_cast<double>(RRCA::factorial(k)));
//         
//         for(auto i = 0; i < n; ++i){//compute for s_j=zeta_j
//             Vector s = zeta.col(i);
//             for(auto j = 0; j < n; ++j){
//                 Vector zs = zeta.col(j);
//                 double prod(oneoverk);
//                 for(auto l = 0; l < d; ++l){
//                     prod *=  (zs(l) - s(l)>=0.0 ? pow(zs(l) - s(l),k) : 0.0);
//                 }
//                 LPM(i) +=  prod * hvec(j);
//             }
//         }
//         
//         //      now integrate over the same thing setting h(z) = 1/n
//         for(auto i = 0; i < n; ++i){//compute for s_j=zeta_j
//             Vector s = zeta.col(i);
//             for(auto j = 0; j < n; ++j){
//                 Vector zs = zeta.col(j);
//                 double prod(oneoverk);
//                 for(auto l = 0; l < d; ++l){
//                     prod *=  (zs(l) - s(l)>=0 ? pow(zs(l) - s(l),k) : 0);
//                 }
//                 LPM(i) +=  prod * hvec(j);
//             }
//         }
//         
// 
//         
//     }

//                       /*
//     *    \brief computes the conditional MCRPS score from Gneiting and Raftery (2007), use monte carlo intergration
//     */
    double conditionalMCRPS(bool structuralYes, const Eigen::MatrixXd& testXX, const Eigen::MatrixXd& testYY ){
//         compute distribution function for yint. this fixes the quantiles over which we integrate
//      simulate normal
        std::mt19937 generator;
        std::normal_distribution<double> dist;
        auto norm = [&] () {
            return dist ( generator );
        };
        const unsigned int howMany(1000);
        Matrix z = RRCA::Matrix::NullaryExpr ( testYY.rows(),howMany, norm );
//      now create howMany x yint.size() matrix of indicators
        
        Matrix fxmat(howMany,yint.cols());
        for(auto i = 0; i < howMany; ++i){
            Matrix check = (yint.colwise()- z.col(i)).unaryExpr([](double elem) { return elem <= 0.0 ? 1.0 : 0.0; });
            fxmat.row(i) = check.colwise().prod();
        }
//      Fx is the howMany x testy.cols() matrix of distribution functions 
        const Matrix Fx = condExpfY_X ( fxmat,testXX,structuralYes ); 
        
        
        Matrix indicatormat( howMany,testYY.cols());
        for(auto i = 0; i < testYY.cols(); ++i){
            Matrix check = (z.colwise()- testYY.col(i)).unaryExpr([](double elem) { return elem >= 0.0 ? 1.0 : 0.0; });
            indicatormat.col(i) = check.colwise().prod().transpose();
        }
        const Matrix integrator = (Fx-indicatormat).array().pow(2);

        return (integrator.mean());
    }
    
    double getasymteststat(double lam,double l, double prec, bool structuralYes){ 
        nt = ((!static_cast<bool>(xdata.cols() % 3) ? xdata.cols()/3 : (!static_cast<bool>((xdata.cols()-1) % 3) ? (xdata.cols()-1)/3 : (xdata.cols()-2)/3 ) ));
        
        Matrix tzdat (zeta.rows(),nt);
        Matrix tzetadat(zeta.rows(),nt);
        double const ntd(static_cast<double>(nt));
        
        tzetadat.topRows(xdata.rows())     = xdata(Eigen::all, Eigen::seqN(0, nt, 2));
        tzetadat.bottomRows(ydata.rows())  = ydata(Eigen::all, Eigen::seqN(1, nt, 2) );
        
        tzdat.topRows(xdata.rows())        = xdata.rightCols(nt);
        tzdat.bottomRows(ydata.rows())     = ydata.rightCols(nt);
        
        
//      need to introduce new pivoted Cholesky here in order to compute with respect to iid product measure
        KernelMatrix Kzeta_iid(tzetadat);
        LowRank pivzeta_iid;
        KernelBasis baszeta_iid(Kzeta_iid,pivzeta_iid);

        Kzeta_iid.kernel().l = l;
        
        pivzeta_iid.compute ( Kzeta_iid, prec );
        baszeta_iid.init(Kzeta_iid, pivzeta_iid.pivots());
        baszeta_iid.initNewtonBasisWeights(pivzeta_iid);
        Matrix Bzeta_iid = baszeta_iid.matrixU();
        Matrix Lzeta_iid = baszeta_iid.eval(tzetadat) * Bzeta_iid;
        unsigned int m = Lzeta_iid.cols();
        
        

        Matrix zeta_zs = basz.eval(tzetadat);
        Matrix zeta_zetas = baszeta.eval(tzetadat);
        
        Vector hvec;
        if(structuralYes){
            hvec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec).cwiseMax(-p);
        } else {
            hvec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec);
        }
        // std::cout << " L'L " << std::endl << Lzeta_iid.transpose() * Lzeta_iid << std::endl;
        
        Matrix G_n = Lzeta_iid.transpose()*Lzeta_iid;
        Vector outerHelp = Lzeta_iid.colwise().sum().transpose();
        Eigen::SelfAdjointEigenSolver<RRCA::Matrix> es (2.0/(ntd*ntd)*(G_n - outerHelp*outerHelp.transpose()/ntd));
        // Eigen::BDCSVD<RRCA::Matrix> svd(2.0/ntd*(G_n - outerHelp*outerHelp.transpose()/ntd),Eigen::ComputeThinU | Eigen::ComputeThinV);
        // svd.setThreshold(1.e-13);
        std::cout << "eigenvalues " << es.eigenvalues().transpose() << std::endl;
        Vector need = sqrt(ntd) * (Lzeta_iid/ntd+lam * Bzeta_iid).transpose() * hvec;
        // std::cout << "projection " << (Bzeta_iid.transpose() * hvec).transpose() << std::endl;
        std::cout << "need " << need << std::endl;
        std::cout << " testvec " << ((es.eigenvectors().rightCols(m-1).transpose() * need).array().square().matrix().cwiseQuotient(es.eigenvalues().tail(m-1))) << std::endl;
        double teststat(((es.eigenvectors().rightCols(m-1).transpose() * need).array().square().matrix().cwiseQuotient(es.eigenvalues().tail(m-1))).sum());
        // for(int i = 1; i < m ; ++i){
        //     std::cout << i << '\t' << es.eigenvectors().col(i).dot(need)/ es.eigenvalues()(i) << std::endl;
        //     teststat += pow(es.eigenvectors().col(i).dot(need),2)/ es.eigenvalues()(i);
        // }
        // for(int i = 0; i < m-1 ; ++i){
        //     std::cout << i << '\t' << svd.matrixU().col(i).dot(need)/ svd.singularValues()(i) << std::endl;
        //     teststat += pow(svd.matrixU().col(i).dot(need),2)/ svd.singularValues()(i);
        // }
        nt = m-1;
        
        return(teststat);
    }
    
        double getmcpval(double lam,double l, bool structuralYes){ 
        nt = ((!static_cast<bool>(xdata.cols() % 3) ? xdata.cols()/3 : (!static_cast<bool>((xdata.cols()-1) % 3) ? (xdata.cols()-1)/3 : (xdata.cols()-2)/3 ) ));
        
        Matrix tzdat (zeta.rows(),nt);
        Matrix tzetadat(zeta.rows(),nt);
        double const ntd(static_cast<double>(nt));
        
        tzetadat.topRows(xdata.rows())     = xdata(Eigen::all, Eigen::seqN(0, nt, 2));
        tzetadat.bottomRows(ydata.rows())  = ydata(Eigen::all, Eigen::seqN(1, nt, 2) );
        
        tzdat.topRows(xdata.rows())        = xdata.rightCols(nt);
        tzdat.bottomRows(ydata.rows())     = ydata.rightCols(nt);
        
        KernelMatrix Ktestzeta(tzetadat);
        KernelBasis  bastzeta;
        bastzeta.initFull(Ktestzeta);
        Matrix K = bastzeta.matrixKpp();
        
        

        Matrix zeta_zs = basz.eval(tzetadat);
        Matrix zeta_zetas = baszeta.eval(tzetadat);
        
        Vector hvec;
        if(structuralYes){
            hvec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec).cwiseMax(-p);
        } else {
            hvec = (zeta_zetas * u_til_lambda +  zeta_zs * lambda_vec);
        }
        
        Vector outerHelp = K.colwise().sum().transpose();
        Eigen::SelfAdjointEigenSolver<RRCA::Matrix> es ( 2.0/(ntd*ntd)*(K- outerHelp*outerHelp.transpose()/ntd) );
        
        std::cout << es.eigenvalues().transpose() << std::endl;
        
        Vector need = (1.0/ntd * K+lam * Matrix::Identity(nt,nt)) * hvec;
        
//      now find the index of the eigenvectors that is greater or equal than 1e-13
        auto upper = std::upper_bound(es.eigenvalues().begin(), es.eigenvalues().end(), 1.e-16);
        const bool upperfound = upper != es.eigenvalues().end() && *upper >= 1.e-16;
        const unsigned int tind = es.eigenvalues().size() - std::distance(es.eigenvalues().begin(), upper)-1;
        // std::cout << "tind " << tind << std::endl;
        
        Vector diagEs = es.eigenvalues().tail(tind);
        
        double tstat = (es.eigenvectors().rightCols(tind).transpose() * need).squaredNorm();
        // std::cout << "tstat " << tstat << std::endl;
        
        std::mt19937 generator;
        std::normal_distribution<double> dist;
        auto norm = [&] () {
            return dist ( generator );
        };
        const unsigned int howMany(10000);
        Vector samples = (RRCA::Matrix::NullaryExpr (tind,howMany, norm ).array().square().colwise()*diagEs.array()).colwise().sum().transpose();
//      now sort the samples ascendingly
        
        std::sort(samples.begin(), samples.end() ,std::less<double>() );
        // std::cout << " samples " << samples.transpose() << std::endl;
//          now find the index 
        auto lower = std::upper_bound(samples.begin(), samples.end(), tstat);
        // check that value has been found
        const bool found = lower != samples.end() && *lower >= tstat;
        // std::cout << "tind 2 " << std::distance(samples.begin(), lower) << std::endl;
        if(found){
            return(static_cast<double>(std::distance(samples.begin(), lower))/static_cast<double>(howMany));
        }
        
        return(1.0);
    }
    
    unsigned int getNt() const {
        return(nt);
    }

    

private:
    const Matrix xdata;
    const Matrix ydata;
    const unsigned int n;
    Matrix zdata;
    
    Matrix y_z;
    
    Matrix zeta;
    
    Matrix yzeta;
    Matrix xzeta;
    
    Matrix Lzeta; // the important ones
    Matrix Lz; // the important ones


    Matrix Bzeta; // the basis transformation matrix
    Matrix Bz; // the basis transformation matrix for the q lambda projector
    Vector q_til_lambda;
    Vector u_til_lambda;
    Matrix toInvert;
    Matrix L_tL;
    
    Vector lambda_vec;
    iVector iid_int; // the indices for the iid integration
    Matrix yint; // to compute the conditional expectations. iid sample
//     Matrix Q;
    
    // Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
    // Vector prob_vec; // the important ones
    unsigned int nt;

    KernelMatrix Kzeta;
    LowRank pivzeta;
    KernelBasis baszeta;
 
    KernelMatrix Kz;
    LowRank pivz;
    KernelBasis basz;
    

    
    double tol;
    static constexpr double p = 1.0;
    
    
    
//     these are the empirical bounds for the functions

//     /*
//    *    \brief computes the kernel basis and tensorizes it
//    */ 
    void precomputeKernelMatrices ( double l, double prec, double lambda){
        Kzeta.kernel().l = l;
        Kz.kernel().l = l;
        
        pivzeta.compute ( Kzeta, prec );
        baszeta.init(Kzeta, pivzeta.pivots());
        baszeta.initNewtonBasisWeights(pivzeta);
        Bzeta = baszeta.matrixU();
        Lzeta = baszeta.eval(zeta) * Bzeta;
        
        pivz.compute ( Kz, prec );
        basz.init(Kz,pivz.pivots());
        basz.initNewtonBasisWeights(pivz);
        Bz = basz.matrixU();
        Lz = basz.eval(zdata) * Bz;
               
        lambda_vec = Bz  *  Lz.colwise().mean().transpose()/ lambda;
        
        q_til_lambda = basz.eval(zeta) * lambda_vec;
    }
};


template<typename KernelMatrix, typename LowRank, typename KernelBasis>
constexpr double DistributionEmbeddingSample<KernelMatrix, LowRank, KernelBasis>::p;















} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
