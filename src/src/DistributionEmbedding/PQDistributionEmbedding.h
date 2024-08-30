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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_PQDISTRIBUTIONEMBEDDING_H_
#define RRCA_DISTRIBUTIONEMBEDDING_PQDISTRIBUTIONEMBEDDING_H_




#include <gsl/gsl_multimin.h>

namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{


template<typename KernelMatrix, typename LowRank, typename KernelBasis>
class PQDistributionEmbedding
{
    typedef typename KernelMatrix::value_type value_type;



public:
//  data_ contains both the q and the p data. we always assume that we have the same number of samples
//  both the q and the p data are potentially decomposed in x and y coordinates
//  p data are the first n cols and q data are the last n cols
//     if ydim = 0, they are not decomposed
    PQDistributionEmbedding ( const Matrix& data_, const Matrix& val_data_ ,const Vector& rowStds_, unsigned int ydim_ = 0) :
        data(data_),
        rowStds(rowStds_),
        stdData(data_.array().colwise()/rowStds_.array()),
        pdata ( data_.leftCols(data_.cols()/2)),
        intdata_y ( data_.bottomRows(ydim_)),
        qdata ( data_.rightCols(data_.cols()/2)),
        val_data(val_data_),
        val_pdata ( val_data_.leftCols(val_data.cols()/2)),
        val_qdata ( val_data_.rightCols(val_data.cols()/2)),
        ydim(ydim_),
        n(data_.cols()/2),
        nv(val_data_.cols()/2),
        K ( stdData ),
        bas(K,piv) {
            assert(data_.cols() % 2 == 0 );
            assert(val_data_.cols() % 2 == 0 );
            assert(data_.rows()  == val_data_.rows()  );
        }

        const Vector& getH() const {
            return(h);
        }




      /*
   *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
   */
    int solve(const std::vector<double>& parms){
        for(unsigned int i = 0; i < parms.size()-1; ++i){ // last parm is always lambda
            K.kernel().setParameter(parms[i],i);
        }
        double const lambda(parms.back());


        piv.compute ( K,RRCA_LOWRANK_EPS);
        bas.init(K, piv.pivots());
        bas.initNewtonBasisWeights(piv);
        U =  bas.matrixU() ;

        L = bas.eval ( stdData ) * U;
        m = U.cols();

//      assuming that p=1
        h = U * (L.topRows(n).transpose() * L.topRows(n) + n * lambda * Matrix::Identity(m,m)).llt().solve( L.bottomRows(n).colwise().sum().transpose()-L.topRows(n).colwise().sum().transpose());

        return(EXIT_SUCCESS);
    }



    /*
   *    \brief this is the L2 loss function to be evaluated at the validation or test data
   */
    double validationScore(double lambda) const{
        const Matrix valk = bas.eval((val_data.array().colwise()/rowStds.array()).matrix()); // valn \times m
        double const score((h.size() < nv ? nv*lambda * h.squaredNorm() : 0.0) + (valk.topRows(nv)*h).squaredNorm() + 2*(valk.topRows(nv).colwise().sum() - valk.bottomRows(nv).colwise().sum() ) *h);
        // double const score((valk.topRows(nv)*h).squaredNorm() + 2*(valk.topRows(nv).colwise().sum() - valk.bottomRows(nv).colwise().sum() ) *h);
        return(score);
    }

     /*
   *    \brief this is the L2 loss function t performs k-fold time series cross validation over the trainings sample
   */
    double cross_validationScore(const std::vector<double>& parms, double prec = RRCA_LOWRANK_EPS, unsigned int k = 5) const{
        double sum ( 0 );

        double const lambda(parms.back());
        Eigen::VectorXi  bigseq = Eigen::VectorXi::LinSpaced ( n,0,n-1 );
        Eigen::VectorXi  ind = Eigen::VectorXi::LinSpaced ( n/k,0,n-1 );
        std::set<int> uniqueq{ind.begin(), ind.end() };
        std::set<int>::iterator it = uniqueq.end();
            --it;
        if((*it) == n-1) uniqueq.erase(std::prev(uniqueq.end()));
        std::vector<int> oida{uniqueq.rbegin(), uniqueq.rend()};
        std::sort(oida.begin(), oida.end());
        // for(const auto& itt : oida){
        //     std::cout << itt << '\t';
        // }
        // std::cout <<  std::endl;

        #pragma omp parallel for reduction(+ : sum)
        for ( auto i : oida ) {
            unsigned int endind = std::min ( i+k-1,n-1 );
            // auto notseq = Eigen::seq ( i,endind );
            RRCA::iVector notsmall = RRCA::iVector::LinSpaced(endind-i+1, i,endind);
            RRCA::iVector bignotseq(2*(endind-i+1));
            bignotseq << notsmall, notsmall.array()+n;
            Eigen::VectorXi seq1 = Eigen::VectorXi::LinSpaced ( i,0,i-1 );
            unsigned int endendind = std::min ( endind+1,n-1 );

            Eigen::VectorXi seq2 = Eigen::VectorXi::LinSpaced ( n-1- ( endendind-1 ),endendind,n-1 );
            Eigen::VectorXi seq ( 2*(seq1.size()+seq2.size()) );
            seq << seq1, seq2, seq1.array()+n, seq2.array()+n;
//             since we have P in the first half and Q in the second half we can add the sequence + n to have a proper trainings and validation sample

            // std::cout << " seq " << seq.transpose() << std::endl;
            // std::cout << " notseq " <<  bignotseq.transpose() << std::endl;


            const Matrix train_cross    = data ( Eigen::all, seq );
            const Matrix eval_cross     = data ( Eigen::all,bignotseq );

            auto mod = make_unique<PQDistributionEmbedding<KernelMatrix,LowRank,KernelBasis> >(train_cross,eval_cross, rowStds,ydim);
            mod->solve(parms); // defined in cmake file

            sum += mod->validationScore(lambda);
            // std::cout << mod->validationScore(lambda) << std::endl;
        }
        return ( sum );
    }


            /*
   *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
   */
    Matrix condExpfVec ( const Matrix& Xs, bool structuralYes = true ) const {
        const unsigned int xdim(data.rows()-ydim);
        assert(xdim == Xs.rows());

        Matrix out(intdata_y.cols(),Xs.cols());

//      we use the y portfion of P as integration variables
//      we have first the X coordinates and then the Y coordinates. Z=(X,Y)
        const unsigned int howMany(Xs.cols());
        
        Matrix arg(data.rows(), intdata_y.cols());
        arg.bottomRows(ydim) = intdata_y;
        
        for(unsigned int i = 0; i < howMany; ++i){
            arg.topRows(xdim) = Xs.col(i).replicate(1,intdata_y.cols());
            out.col(i) = bas.eval((arg.array().colwise()/rowStds.array()).matrix()) * h + Vector::Ones(intdata_y.cols());
        }

        if(structuralYes){
            return(out.array().cwiseMax(0.0).rowwise()/out.cwiseMax(0.0).colwise().sum().array());
        }

        return (out/static_cast<double>(intdata_y.cols()));
    }


    const Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs, bool structuralYes = false  ) const {
        return(Ys * condExpfVec(Xs,structuralYes));
    }


    double transformParm(double val, unsigned int which) const {
        return(K.kernel().parmTransform(val, which));
    }

    static double static_my_func(const gsl_vector* v, void* params) // proxy function
    {
        PQDistributionEmbedding<KernelMatrix, LowRank, KernelBasis>* obj = static_cast<PQDistributionEmbedding<KernelMatrix, LowRank, KernelBasis>*>(params);
        std::vector<double> parms(v->size);
        for(unsigned int i = 0; i < v->size-1; ++i){
            parms[i] = obj->transformParm(gsl_vector_get(v, i), i);
        }
        parms[v->size-1] = exp(gsl_vector_get(v, v->size-1));

        obj->solve(parms); // defined in cmake file
        return(obj->validationScore(parms[v->size-1]));
    }

    static double static_my_cross_func(const gsl_vector* v, void* params) // proxy function
    {
        PQDistributionEmbedding<KernelMatrix, LowRank, KernelBasis>* obj = static_cast<PQDistributionEmbedding<KernelMatrix, LowRank, KernelBasis>*>(params);
        std::vector<double> parms(v->size);
        for(unsigned int i = 0; i < v->size-1; ++i){
            parms[i] = obj->transformParm(gsl_vector_get(v, i), i);
        }
        parms[v->size-1] = exp(gsl_vector_get(v, v->size-1));
        return(obj->cross_validationScore(parms));
    }
    std::vector<double> gsl_validate_multistart(bool CROSSVAL=false, unsigned int howmanystarts = 20) {
        const unsigned int KERNPARMNUM = K.kernel().PARMNUM;
        const unsigned int PARMNUM =KERNPARMNUM + 1; // with lambda

        std::mt19937 gen;  //here you could also set a seed
        std::normal_distribution<double> normdist;
        auto norm = [&] () {
            return normdist ( gen );
        };

        double bestF(1e16);
        std::vector<double> bestParms(PARMNUM);



        for(unsigned int iii = 0; iii < howmanystarts; ++iii){
//          draw starting values
            Vector startParms = Vector::NullaryExpr(PARMNUM,norm);
            startParms(PARMNUM-1) = -std::abs(startParms(PARMNUM-1));
            std::cout << "startparms " << startParms.transpose() << std::endl;


            const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
            gsl_multimin_fminimizer *s = NULL;
            gsl_vector *ss, *xx;
            gsl_multimin_function minex_func;

            size_t iter = 0;
            int status;
            double size;
            /* Starting point */
//          the last is always lambda
            xx = gsl_vector_alloc (PARMNUM);
            for(unsigned int i = 0; i < KERNPARMNUM; ++i){
                gsl_vector_set (xx, i, startParms(i));
            }

            gsl_vector_set (xx, KERNPARMNUM, startParms(KERNPARMNUM));


            /* Set initial step sizes to 1 */
            ss = gsl_vector_alloc (PARMNUM);
            gsl_vector_set_all (ss, 1.0);

            /* Initialize method and iterate */
            minex_func.n = PARMNUM;
            minex_func.f = (CROSSVAL ? 
                            &RRCA::DISTRIBUTIONEMBEDDING::PQDistributionEmbedding<KernelMatrix, LowRank, KernelBasis>::static_my_cross_func : 
                            &RRCA::DISTRIBUTIONEMBEDDING::PQDistributionEmbedding<KernelMatrix, LowRank, KernelBasis>::static_my_func); 
            minex_func.params = this;

            s = gsl_multimin_fminimizer_alloc (T, PARMNUM);
            gsl_multimin_fminimizer_set (s, &minex_func, xx, ss);

            do {
                iter++;
                status = gsl_multimin_fminimizer_iterate(s);

                if (status) break;

                size = gsl_multimin_fminimizer_size (s);
                status = gsl_multimin_test_size (size, 1e-2);

                if (status == GSL_SUCCESS) {
                    printf ("converged to minimum at\n");
                    for( unsigned int i = 0; i < KERNPARMNUM; ++i){
                        std::cout << K.kernel().parmTransform(gsl_vector_get (s->x, i), i) << '\t';
                        bestParms[i]   = K.kernel().parmTransform(gsl_vector_get (s->x, i), i);
                    }
                    bestParms[KERNPARMNUM]   = exp(gsl_vector_get (s->x, KERNPARMNUM)); //lambda
                    std::cout << exp(gsl_vector_get (s->x, KERNPARMNUM)) << " f val " << s->fval << std::endl;
                }


                // std::cout << " f val " << s->fval << std::endl;
                if(s->fval < bestF){
                    bestF       = s->fval;
                    std::cout << iter << " Parameter ";
                    for( unsigned int i = 0; i < KERNPARMNUM; ++i){
                        std::cout << K.kernel().parmTransform(gsl_vector_get (s->x, i), i) << '\t';
                        bestParms[i]   = K.kernel().parmTransform(gsl_vector_get (s->x, i), i);
                    }
                    bestParms[KERNPARMNUM]   = exp(gsl_vector_get (s->x, KERNPARMNUM)); //lambda
                    std::cout << exp(gsl_vector_get (s->x, KERNPARMNUM)) << " f val " << s->fval << std::endl;

                }
            } while (status == GSL_CONTINUE && iter < 100);

            gsl_vector_free(xx);
            gsl_vector_free(ss);
            gsl_multimin_fminimizer_free (s);
        }


            return(bestParms);
    }

//  gsl stuff defined in Macros.h
    std::vector<double> gsl_validate(bool CROSSVAL=false) {
            const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
            gsl_multimin_fminimizer *s = NULL;
            gsl_vector *ss, *xx;
            gsl_multimin_function minex_func;

            size_t iter = 0;
            int status;
            double size;
            const unsigned int KERNPARMNUM = K.kernel().PARMNUM;
            const unsigned int PARMNUM =KERNPARMNUM + 1; // with lambda
            /* Starting point */
//          the last is always lambda
            xx = gsl_vector_alloc (PARMNUM);
            for(unsigned int i = 0; i < KERNPARMNUM; ++i){
                gsl_vector_set (xx, i, K.kernel().parmInverseTransform(0.9, i));
            }

            gsl_vector_set (xx, KERNPARMNUM, log(0.00001));


            /* Set initial step sizes to 1 */
            ss = gsl_vector_alloc (PARMNUM);
            gsl_vector_set_all (ss, 1.0);

            /* Initialize method and iterate */
            minex_func.n = PARMNUM;
            minex_func.f = (CROSSVAL ?
                                &RRCA::DISTRIBUTIONEMBEDDING::PQDistributionEmbedding<KernelMatrix, LowRank, KernelBasis>::static_my_cross_func :
                                &RRCA::DISTRIBUTIONEMBEDDING::PQDistributionEmbedding<KernelMatrix, LowRank, KernelBasis>::static_my_func);
            minex_func.params = this;

            s = gsl_multimin_fminimizer_alloc (T, PARMNUM);
            gsl_multimin_fminimizer_set (s, &minex_func, xx, ss);

            do {
                iter++;
                status = gsl_multimin_fminimizer_iterate(s);

                if (status) break;

                size = gsl_multimin_fminimizer_size (s);
                status = gsl_multimin_test_size (size, 1e-2);

                if (status == GSL_SUCCESS) {
                    printf ("converged to minimum at\n");
                }

                std::cout << iter << " Parameter ";
                for( unsigned int i = 0; i < KERNPARMNUM; ++i){
                    std::cout << K.kernel().parmTransform(gsl_vector_get (s->x, i), i) << '\t';
                }
                std::cout << exp(gsl_vector_get (s->x, KERNPARMNUM)) << " f val " << s->fval << std::endl;
            } while (status == GSL_CONTINUE && iter < 100);

            std::vector<double> res(PARMNUM);
            for(unsigned int i = 0; i < KERNPARMNUM; ++i){
                res[i] =  K.kernel().parmTransform(gsl_vector_get (s->x, i), i);
            }
            res[KERNPARMNUM] = exp(gsl_vector_get (s->x, KERNPARMNUM)); // this is lambda


            gsl_vector_free(xx);
            gsl_vector_free(ss);
            gsl_multimin_fminimizer_free (s);



            return(res);
        }

        const Matrix& getYint() const {
            return(intdata_y);
        }

//         \brief outputs a vector containing all teststatistics up to m
        Vector testStatisticVector() const {
            const double nnn(n);
            const Matrix Sigma = L.bottomRows(n).transpose() * L.bottomRows(n)/nnn
                                -L.bottomRows(n).colwise().sum().transpose()*L.bottomRows(n).colwise().sum()/(nnn*nnn)
                                +L.topRows(n).transpose() * L.topRows(n)/nnn
                                -L.topRows(n).colwise().sum().transpose()*L.topRows(n).colwise().sum()/(nnn*nnn);
//                                 smallest first because eigen solver puts smallest eigenvalues first
            const Vector v_lam = ((L.bottomRows(n).colwise().sum().transpose() - L.topRows(n).colwise().sum().transpose()))/sqrt(nnn);

            Eigen::SelfAdjointEigenSolver<Matrix> es(Sigma);
            const Vector esval = es.eigenvalues().reverse();
            const Matrix esvec = es.eigenvectors().rowwise().reverse();
            Vector testStatVec = ((esvec.transpose() * v_lam).array().square()/esval.array());
            Vector sumTestStatVec(testStatVec.size());
            std::partial_sum(testStatVec.cbegin(), testStatVec.cend(), sumTestStatVec.begin(), std::plus<double>());


            return(sumTestStatVec);
        }




private:
    const Matrix& data;
    const Vector rowStds;
    const Matrix stdData;
    const Matrix pdata;
    const Matrix intdata_y;
    const Matrix qdata;

    const Matrix& val_data;
    const Matrix val_pdata;
    const Matrix val_qdata;

    const unsigned int ydim;
    const unsigned int n;
    const unsigned int nv;
    unsigned int m;

    Vector h; // the vector of coefficients

    Matrix L;
    Matrix U; // the basis transformation matrix



    KernelMatrix K;
    LowRank piv;

    KernelBasis bas;

    static constexpr double p = 1.0;




};








} // namespace DISTRIBUTIONEMBEDDING

}  // namespace RRCA
#endif
