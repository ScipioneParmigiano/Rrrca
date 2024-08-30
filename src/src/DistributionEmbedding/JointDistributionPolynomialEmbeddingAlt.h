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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_JOINTDISTRIBUTIONPOLYNOMIALEMBEDDINGALT_H_
#define RRCA_DISTRIBUTIONEMBEDDING_JOINTDISTRIBUTIONPOLYNOMIALEMBEDDINGALT_H_



namespace RRCA {
namespace DISTRIBUTIONEMBEDDING {

/*
*    \brief specializes joint ditribtion embedding to polynomials. here g=1+h
*/
template<unsigned int order>
class JointPolynomialDistributionEmbeddingAlt {
public:
    JointPolynomialDistributionEmbeddingAlt ( const Matrix& xdata_, const Matrix& ydata_) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        xbasisdim ( binomialCoefficient ( xdata_.rows()+order,xdata_.rows() ) ), // no linear terms and no constant
        ybasisdim ( binomialCoefficient ( ydata_.rows()+order,ydata_.rows() ) ), // no linear terms and no constant
        xIndex ( xdata_.rows(),order ),
        yIndex ( ydata_.rows(),order ),
        Vx_t ( xbasisdim,xdata_.cols() ),
        Vy_t ( ybasisdim,ydata_.cols() ),
        modelHasBeenSet(false){
//       compute Vandermonde matrices
        const auto &myxSet = xIndex.get_MultiIndexSet();
        for ( const auto &ind1 : myxSet ) {
            // if(ind1.sum() > 1) // no constant and no linear terms
                xinter.push_back ( ind1 );
        }

        for ( unsigned int i = 0; i < xdata.cols(); ++i ) {
//          compute basisvector
            for ( auto j = 0; j < xinter.size(); ++j ) { // for moments 0 to U.cols()
                double accum = 1;
                iVector ind =xinter[j];
                for ( auto k = 0; k < xdata.rows(); ++k ) {
                    accum *= std::pow ( xdata ( k, i ), ind ( k ) );
                }
                Vx_t ( j,i ) = accum;
            }
        }

        const auto &myySet = yIndex.get_MultiIndexSet();
        for ( const auto &ind1 : myySet ) {
            // if(ind1.sum() > 1) // no constant and no linear terms
                yinter.push_back ( ind1 );
        }

        for ( unsigned int i = 0; i < ydata.cols(); ++i ) {
//          compute basisvector
            for ( auto j = 0; j < yinter.size(); ++j ) { // for moments 0 to U.cols()
                double accum = 1;
                iVector ind =yinter[j];
                for ( auto k = 0; k < ydata.rows(); ++k ) {
                    accum *= std::pow ( ydata ( k, i ), ind ( k ) );
                }
                Vy_t ( j,i ) = accum;
            }
        }
        

//      compute the V matrices
        precomputeKernelMatrices();
    }
    ~JointPolynomialDistributionEmbeddingAlt(){
        if(modelHasBeenSet){
#ifdef RRCA_HAVE_MOSEK
            M->dispose();
#endif
        }
    }
    
    unsigned int getSubspaceDimension() const {
        return(Vx_t.rows()+Vy_t.rows());
    }
    const Matrix& getH() const {
        return ( H );
    }
    
    double validationScore(const Matrix& Xs, const Matrix& Ys) const{
        Matrix VVx_t(xbasisdim,Xs.cols() );
        Matrix VVy_t(ybasisdim,Ys.cols() );
        for ( unsigned int i = 0; i < Xs.cols(); ++i ) {
//          compute basisvector
            for ( auto j = 0; j < xinter.size(); ++j ) { // for moments 0 to U.cols()
                double accum = 1;
                iVector ind =xinter[j];
                for ( auto k = 0; k < Xs.rows(); ++k ) {
                    accum *= std::pow ( Xs ( k, i ), ind ( k ) );
                }
                VVx_t ( j,i ) = accum;
            }
        }

        for ( unsigned int i = 0; i < Ys.cols(); ++i ) {
//          compute basisvector
            for ( auto j = 0; j < yinter.size(); ++j ) { // for moments 0 to U.cols()
                double accum = 1;
                iVector ind =yinter[j];
                for ( auto k = 0; k < Ys.rows(); ++k ) {
                    accum *= std::pow ( Ys ( k, i ), ind ( k ) );
                }
                VVy_t ( j,i ) = accum;
            }
        }
        
        
        const double n(Xs.cols());
        // std::cout << "Xs: " << Xs.topLeftCorner(2,2) << std::endl;
        // std::cout << "Ys: " << Ys.topLeftCorner(2,2) << std::endl;

         Matrix oas =  (-  VVy_t.transpose() * ygramL * H * xgramL * VVx_t - Matrix::Constant(Ys.cols(),Ys.cols(), 1.0))/n;
         
         oas.diagonal() += Vector::Constant(Ys.cols(),1);
         // oas.array() -= 1.0;
//          std::cout << "oas " << oas.topLeftCorner(3,3) << std::endl;
//          
//          std::cout << "oas min max" << oas.maxCoeff() << '\t' << oas.minCoeff()  << std::endl;

        
        return((oas).squaredNorm());
    }


    /*
    *    \brief solves the finite-dimensional optimization problem for given kernel parameterization
    */
    int solve(double lam = 0){
#ifdef RRCA_HAVE_MOSEK
            return(solveMosek(lam));
#endif
        return(EXIT_FAILURE);
    }




//
    /*
    *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
    */
    int solveUnconstrained ( double lam = 0.0) {
        H = helper/((1.0+lam));
        // std::cout << "H" << H << std::endl;
        
        return ( EXIT_SUCCESS );
    }


    /*
    *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
    */
    Matrix condExpfVec ( const Matrix& Xs ) const {
//      compute first the new Vandermonde using Xs
        const double n ( Qx.rows() );
        Matrix newVandermonde_t ( xinter.size(),Xs.cols() );
        for ( unsigned int i = 0; i < Xs.cols(); ++i ) {
//          compute basisvector
            for ( auto j = 0; j < xinter.size(); ++j ) { // for moments 0 to U.cols()
                double accum = 1;
                RRCA::iVector ind =xinter[j];
                for ( auto k = 0; k < Xs.rows(); ++k ) {
                    accum *= std::pow ( Xs ( k, i ), ind ( k ) );
                }
                newVandermonde_t ( j,i ) = accum;
            }
        }
//      now compute the kernel matrix block
        
        // const Matrix oas = H * ( xgramL * newVandermonde_t );
        const Matrix res = Qy * H * ( xgramL * newVandermonde_t ) + Matrix::Constant(Qy.rows(),Xs.cols(),1 );
        const Matrix resres = res.array().rowwise() /res.colwise().sum().array();
        
        

        return ( resres );
    }

    /*
    *    \brief computes conditional expectation of all the rows in Ys
    */
    const Matrix condExpfY_X ( const Matrix& Ys, const Matrix& Xs ) const {
        const double n ( Qx.rows() );
        Matrix newVandermonde_t ( xinter.size(),Xs.cols() );
        for ( unsigned int i = 0; i < Xs.cols(); ++i ) {
//          compute basisvector
            for ( auto j = 0; j < xinter.size(); ++j ) { // for moments 0 to U.cols()
                double accum = 1;
                RRCA::iVector ind =xinter[j];
                for ( auto k = 0; k < Xs.rows(); ++k ) {
                    accum *= std::pow ( Xs ( k, i ), ind ( k ) );
                }
                newVandermonde_t ( j,i ) = accum;
            }
        }
//      now compute the kernel matrix block
        Matrix oida = ( H * xgramL * newVandermonde_t );
        const RowVector norma = Qy.colwise().sum() * oida + RowVector::Constant(Xs.cols(),Qy.rows() );
        oida.array().rowwise()/=norma.array();
        
        return ( Ys * Qy * oida + (Ys.rowwise().sum().replicate(1,oida.cols()).array().rowwise()/norma.array()).matrix());
        // return(Ys * condExpfVec(Xs));
    }


private:
    const Matrix& xdata;
    const Matrix& ydata;

    const unsigned int xbasisdim;
    const unsigned int ybasisdim;

    const RRCA::MultiIndexSet<RRCA::iVector> xIndex;
    const RRCA::MultiIndexSet<RRCA::iVector> yIndex;

    Matrix Vx_t; // the important ones
    Matrix Vy_t; // the important ones

    Vector h; // the vector of coefficients
    Matrix H; // the matrix of coefficients, h=vec H

    Matrix Qx; // the basis matrix
    Matrix Qy; // the basis transformation matrix

    Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
    Vector prob_vec; // the important ones

    Matrix xgramL;
    Matrix ygramL;
    Matrix helper;


    iVector crossSec;

    std::vector<Index> pivx;
    std::vector<Index> pivy;
    std::vector<Index> piv; // need to select both points

    std::vector<RRCA::iVector> xinter;
    std::vector<RRCA::iVector> yinter;
    
    bool modelHasBeenSet;
#ifdef RRCA_HAVE_MOSEK
    M_Model M;
#endif

    /*
    *    \brief computes the kernel basis and tensorizes it
    */
    void precomputeKernelMatrices (  ) {
//         compute gram matrix

            const double n(Vx_t.cols());
            Eigen::SelfAdjointEigenSolver<Matrix> esx ( Vx_t * Vx_t.transpose()/n );
            xgramL = esx.operatorInverseSqrt();
            
            // std::cout << xgramL << std::endl;

            Qx = Vx_t.transpose() * xgramL; // this is Q r
            
            Eigen::SelfAdjointEigenSolver<Matrix> esy ( Vy_t * Vy_t.transpose()/n );
            ygramL = esy.operatorInverseSqrt();

            Qy = Vy_t.transpose() * ygramL; // this is Q r


        precomputeHelper (  );

    }




    void precomputeHelper ( ) {
        const double n ( Qx.rows() );
        const unsigned int  rankx ( Qx.cols() );
        const unsigned int  ranky ( Qy.cols() );
        const unsigned int  m ( rankx*ranky );
        helper = Qy.transpose() * Qx /n - Qy.colwise().mean().transpose() * Qx.colwise().mean();

        // Vector p_ex_k = helper.reshaped();
        prob_vec = 2.0* ( -  helper.reshaped());
    }


#ifdef RRCA_HAVE_MOSEK
    int solveMosek(double lam) {
        if(!modelHasBeenSet){
            M = new mosek::fusion::Model ( "JointPolynomialDistributionEmbedding" );
            // M->setLogHandler ( [=] ( const std::string & msg ) {
            //     std::cout << msg << std::flush;
            // } );
        // auto _M = monty::finally ( [&]() {
        //     M->dispose();
        // } );

            const unsigned int m ( Qx.cols() *Qy.cols() );
            const double n ( Qx.rows() );



//         can define matrix variable for the bilinear form. this is actually the transpose of HH, because mosek is row major
            M_Variable::t HH_t = M->variable ( "H_t", monty::new_array_ptr<int, 1> ( { ( int ) Qx.cols(), ( int ) Qy.cols() } ), M_Domain::unbounded() );
            M_Variable::t ht = M_Var::flatten ( HH_t );

//      introduce the variable for the PSD constraint
            const M_Matrix::t Gx_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( xgramL.data(), monty::shape ( xgramL.cols(), xgramL.rows() ) ) ) );
            const M_Matrix::t Gy_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( ygramL.data(), monty::shape ( ygramL.cols(), ygramL.rows() ) ) ) );

            M_Variable::t HH_t_bas = M->variable ( "HH_t_bas", monty::new_array_ptr<int, 1> ( { ( int ) Qx.cols(), ( int ) Qy.cols() } ), M_Domain::unbounded() );
        
            M->constraint(M_Expr::sub(HH_t_bas,  M_Expr::mul ( Gx_wrap, M_Expr::mul ( HH_t,Gy_wrap ) )),M_Domain::equalsTo(0.0));
        
            auto multiplier = M->parameter("multiplier");
            multiplier->setValue(1.0+lam);

//         for the quadratic cone
            M_Variable::t uu = M->variable ( "uu",1, M_Domain::greaterThan ( 0.0 ) );

            // auto quadsqrt = std::make_shared<M_ndarray_1> ( monty::shape ( m ), [&] ( ptrdiff_t l ) {
            //     return sqrt ( prob_quadFormMat ( l ) );
            // } );
            auto prob_vecwrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1 ( prob_vec.data(), monty::shape ( m ) ) ) ;
            M->constraint ( "oida", M_Expr::vstack ( 0.5, uu, ht), M_Domain::inRotatedQCone() ); // quadratic cone for objective function


// // // // // // // // // // // // // // // // // //
// // // // // normalization // // // // // // // // 

            Vector LYmeans ( Qy.colwise().mean() );
            Vector LXmeans ( Qx.colwise().mean() );
        
            auto LYmeanswrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1 ( LYmeans.data(), monty::shape ( LYmeans.size() ) ) );
            auto LXmeanswrap = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1 ( LXmeans.data(), monty::shape ( LXmeans.size() ) ) );
        
            M->constraint ( M_Expr::dot ( LXmeanswrap,M_Expr::mul ( HH_t,LYmeanswrap ) ), M_Domain::equalsTo ( 0.0 ) );
        
        // // // // // // // // // // // // // // // // // //
// // // // // normalization // // // // // // // // 
//      positivity constraint

//      embed into polynomial with double order
            const RRCA::MultiIndexSet<RRCA::iVector> myBigIndex ( (xdata.rows() +ydata.rows()),order*2 );
            const RRCA::MultiIndexSet<RRCA::iVector> myBigHalfIndex ( (xdata.rows() +ydata.rows()),order );


            std::vector<iVector> bigInter;
            const auto &myBigSet = myBigIndex.get_MultiIndexSet();
            for ( const auto &ind1 : myBigSet ) {
                bigInter.push_back ( ind1 );
            }

            std::vector<iVector> bigHalfInter;
            const auto &myBigHalfSet = myBigHalfIndex.get_MultiIndexSet();
            for ( const auto &ind1 : myBigHalfSet ) {
                bigHalfInter.push_back ( ind1 );
            }

        
//      find the map from the coefficient vector to the PSD matrix
            std::map<iVector, std::vector<ijVector>, FMCA_Compare<iVector> > bigIndexMap;
//             fill what's needed for the pos def
            for ( int j = 0; j < bigHalfInter.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = bigHalfInter[i] + bigHalfInter[j];
                    ijVector ij;
                    ij << i, j;
                    auto it = bigIndexMap.find ( newIndex );
                    if ( it != bigIndexMap.end() )
                        it->second.push_back ( ij );
                    else
                        bigIndexMap[newIndex].push_back ( ij );
                }
            }

//          now creat another map  to point to the indices in H_t
//          we know from the bilinear form that every combination is unique
            std::map<iVector, ijVector, FMCA_Compare<iVector> > mapMap;
            for ( int i = 0; i < xinter.size(); ++i ) { //these are the x coordinates
                for ( int j = 0; j < yinter.size(); ++j ) { //these are the y coordinates
                    RRCA::iVector newIndex ( xdata.rows() +ydata.rows() );
                    newIndex <<  yinter[j], xinter[i];
                    RRCA::ijVector ij;
                    ij << i, j; // flipped because H_t is transposed
                    mapMap[newIndex] = ij;
                }
            }

        //      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
            M_Variable::t MM = M->variable ( RRCA::M_Domain::inPSDCone ( bigHalfInter.size() ) ); // the moment matrix
//      find the coefficients of the big coefficient vector in the matrix of the quadratic form
            const auto &mySet2 = myBigIndex.get_MultiIndexSet();
            for ( const auto &ind : mySet2 ) {
//          find it in the map
                const std::vector<ijVector> ind1 = bigIndexMap[ind];
                M_Expression::t ee = M_Expr::constTerm ( 0.0 );
                for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
                    if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
                        ee = M_Expr::add ( ee,M_Expr::mul ( 2.0,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) ) );
                    } else {// diagonal
                        ee = M_Expr::add ( ee,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) );
                    }
                }
//          now check whether index is also present in H_t
                auto it = mapMap.find ( ind );
                if ( it != mapMap.end() ) { //present, usual semidefinite constraint
                // std::cout << "setting " << ind.transpose() << "H to i=" << it->second(0 ) << " and j=" << it->second(1) << std::endl;
                    if(ind.sum()>0){ // anything but the constant
                        M->constraint ( M_Expr::sub ( ee,HH_t_bas->index ( it->second ( 0 ),it->second ( 1 ) ) ),RRCA::M_Domain::equalsTo ( 0.0 ) );
                    } else { // the constant must have 1+h=sos
                        M->constraint ( M_Expr::sub ( ee,HH_t_bas->index ( it->second ( 0 ),it->second ( 1 ) ) ),RRCA::M_Domain::equalsTo ( 1.0 ) );
                    }
                }  else { // corresponding coefficient must be zero
                // std::cout << "setting " << ind.transpose() << " coefficient must be zero " << std::endl;
                    M->constraint ( ee,M_Domain::equalsTo ( 0.0 ) );
                }
            }

            M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::add ( M_Expr::mul(multiplier,uu),M_Expr::dot ( prob_vecwrap, ht ) ) );
            modelHasBeenSet = true;
            } else {
                auto multi = M->getParameter("multiplier");
                const double n ( Qx.rows() );
                multi->setValue(1.0+lam);
            }
        
        M->solve();
        
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal ) {
//             const auto htsol = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( * ( ht->level() ) ) );
            const auto soll = M->getVariable("H_t");
            M_ndarray_1 htsol   = * ( soll->level() );
            const Eigen::Map<Matrix> auxmat ( htsol.raw(), Qy.cols(),Qx.cols() );
            H = auxmat;
            h = H.reshaped();
            // std::cout << "H" << H << std::endl;
        } else {
            std::cout << "infeasible  " <<  std::endl;
            return ( EXIT_FAILURE );
        }
        return ( EXIT_SUCCESS );
    }
#endif

};


} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
