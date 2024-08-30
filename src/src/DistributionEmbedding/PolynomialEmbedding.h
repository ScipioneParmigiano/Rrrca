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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_POLYNOMIALEMBEDDING_H_
#define RRCA_DISTRIBUTIONEMBEDDING_POLYNOMIALEMBEDDING_H_



namespace RRCA {
namespace DISTRIBUTIONEMBEDDING {

std::shared_ptr<monty::ndarray<int,1>>    nint ( const std::vector<int> &X ) {
    return monty::new_array_ptr<int> ( X );
}
enum StateSpace {R=0, Rplus=1, Interval=2};




// base class for embedding positive polynomials
class PolynomialEmbedding {
    
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> eigenIndexMatrix;
    typedef Eigen::Matrix<Eigen::Index, 2, 1> eigenIJVector;



public:
    PolynomialEmbedding ( unsigned int d_, unsigned int n_, const Matrix& H_, const StateSpace sp_ ) : d ( d_ ), n ( n_ ),initN ( binomialCoefficient ( n_+d_, d_ ) ), H ( H_ ), sp ( sp_ ) {
        assert ( H.rows() == initN );
        assert ( H.cols() == initN );
    }



    M_Model solveEqualityConstrained ( const Matrix& S, const Vector& rhsS ) {
        
        

        M_Model M = new mosek::fusion::Model ( "EqualityConstrainedPolynomialEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );

        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded() );
        int success;
        if ( sp == StateSpace::Rplus && d ==1 ) {
            success = positivityConstraintPrimalRplus ( M, h );
        } else {
            success = positivityConstraintPrimalRn ( M, h );
        }

        success = fillEqualityConstraints ( M, h, H, S, rhsS );

        success = fillPrimalObjective ( M,h,H );

        return ( M );
    }
//  this is an overridden version for the bounded interval
    M_Model solveEqualityConstrained ( const Matrix& S, const Vector& rhsS, double a, double b ) {
        
        
        assert(sp == StateSpace::Interval);

        M_Model M = new mosek::fusion::Model ( "EqualityConstrainedPolynomialEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );

        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded() );
        int success;
        if ( sp == StateSpace::Interval && d ==1 ) {
            success = positivityConstraintPrimalInterval ( M, h, a, b );
        } else {
            success = positivityConstraintPrimalRn ( M, h );
        }

        success = fillEqualityConstraints ( M, h, H, S, rhsS );

        success = fillPrimalObjective ( M,h,H );

        return ( M );
    }

    M_Model solveInequalityConstrained ( const Matrix& U, const Vector& rhsU ) {
        
        

        M_Model M = new mosek::fusion::Model ( "InequalityConstrainedPolynomialEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );

        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded() );
        int success;
        if ( sp == StateSpace::Rplus && d ==1 ) {
            success = positivityConstraintPrimalRplus ( M, h );
        } else {
            success = positivityConstraintPrimalRn ( M, h );
        }

        success = fillInequalityConstraints ( M, h, H, U, rhsU );

        success = fillPrimalObjective ( M,h,H );

        return ( M );
    }
    
//     this is overriden to account for bounded interval
    M_Model solveInequalityConstrained ( const Matrix& U, const Vector& rhsU, double a, double b ) {
        
        
        assert(sp == StateSpace::Interval);

        M_Model M = new mosek::fusion::Model ( "InequalityConstrainedPolynomialEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );

        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded() );
        int success;
        if ( sp == StateSpace::Interval && d ==1 ) {
            success = positivityConstraintPrimalInterval ( M, h, a, b );
        } else {
            success = positivityConstraintPrimalRn ( M, h );
        }

        success = fillInequalityConstraints ( M, h, H, U, rhsU );

        success = fillPrimalObjective ( M,h,H );

        return ( M );
    }
    

    M_Model solveConstrained ( const Matrix& S, const Vector& rhsS, const Matrix& U, const Vector& rhsU ) {
        
        

        M_Model M = new mosek::fusion::Model ( "ConstrainedPolynomialEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );

        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded() );
        int success;
        if ( sp == StateSpace::Rplus && d ==1 ) {
            success = positivityConstraintPrimalRplus ( M, h );
        } else {
            success = positivityConstraintPrimalRn ( M, h );
        }

        success = fillInequalityConstraints ( M, h, H, U, rhsU );
        success = fillEqualityConstraints ( M, h, H, S, rhsS );

        success = fillPrimalObjective ( M,h,H );

        return ( M );
    }
    
    
    M_Model solveConstrained ( const Matrix& S, const Vector& rhsS, const Matrix& U, const Vector& rhsU,
                                              double a, double b) {
        
        
        assert(sp == StateSpace::Interval);

        M_Model M = new mosek::fusion::Model ( "ConstrainedPolynomialEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );

        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded() );
        int success;
        if ( sp == StateSpace::Interval && d ==1 ) {
            success = positivityConstraintPrimalInterval ( M, h, a, b );
        } else {
            success = positivityConstraintPrimalRn ( M, h );
        }

        success = fillInequalityConstraints ( M, h, H, U, rhsU );
        success = fillEqualityConstraints ( M, h, H, S, rhsS );

        success = fillPrimalObjective ( M,h,H );

        return ( M );
    }

//     only positivity, no additional constraints
    M_Model solveUnconstrained ( double a, double b, bool withPositivity) {
        
        
        assert(sp == StateSpace::Interval);

        M_Model M = new mosek::fusion::Model ( "ConstrainedPolynomialEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );

        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded() );
        int success;
        if(withPositivity){
            if ( sp == StateSpace::Interval && d ==1 ) {
                success = positivityConstraintPrimalInterval ( M, h, a, b );
            } else {
                success = positivityConstraintPrimalRn ( M, h );
            }
        }



//         success = fillPrimalObjective ( M,h,H );

        return ( M );
    }
    
        M_Model solveUnconstrained ( bool withPositivity) {
        
        

        M_Model M = new mosek::fusion::Model ( "ConstrainedPolynomialEmbedding" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );

        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded() );
        int success;
        if(withPositivity){
            if ( sp == StateSpace::Rplus && d ==1 ) {
                success = positivityConstraintPrimalRplus ( M, h );
            } else {
                success = positivityConstraintPrimalRn ( M, h );
            }
        }

//         success = fillPrimalObjective ( M,h,H );

        return ( M );
    }


    







private:
    const  unsigned int d;
    const  unsigned int n;
    const  unsigned int initN;
    const Matrix& H;
    const StateSpace sp;


//
//     template<unsigned int d,unsigned int n>
    int positivityConstraintPrimalRn ( M_Model M, M_Variable::t coeff) const {
        assert ( n % 2 == 0 );
        
        
        const  unsigned int smallN ( binomialCoefficient ( n/2+d, d ) );
        const RRCA::MultiIndexSet<iVector> myIndex ( d,n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
        // std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > indexMap;
        const RRCA::MultiIndexSet<iVector> mySmallIndex ( d,n/2 ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap;

        //  this part is the same for all possibilities
        const auto &mySet = mySmallIndex.get_MultiIndexSet();
//      copy into vector for simplicity
        std::vector<iVector> inter;
        for ( const auto &ind1 : mySet ) {
            inter.push_back ( ind1 );
        }
        for ( int j = 0; j < inter.size(); ++j ) {
            for ( int i = 0; i <= j; ++i ) {
                iVector newIndex = inter[i] + inter[j];
                eigenIJVector ij;
                ij << i, j;
                auto it = smallIndexMap.find ( newIndex );
                if ( it != smallIndexMap.end() ) {
                    it->second.push_back ( ij );
                } else {
                    smallIndexMap[newIndex].push_back ( ij );
                }
            }
        }

        //      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
        M_Variable::t MM = M->variable ("SemiCone1", M_Domain::inPSDCone ( ( int ) smallN ) ); // the moment matrix
//         the coefficients



//      find the coefficients of the big coefficient vector in the matrix of the quadratic form
        const auto &mySet2 = myIndex.get_MultiIndexSet();
        unsigned int counter = 0;
        for ( const auto &ind : mySet2 ) {
//          find it in the map
            const std::vector<eigenIJVector> ind1 = smallIndexMap[ind];
            M_Expression::t ee = M_Expr::constTerm ( 0.0 );
            for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
                if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
                    ee = M_Expr::add ( ee,M_Expr::mul ( 2.0,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) ) );
                } else {// diagonal
                    ee = M_Expr::add ( ee,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) );
                }
            }
            M->constraint ( M_Expr::sub ( ee,coeff->index ( counter ) ),M_Domain::equalsTo ( 0.0 ) );
            ++counter;
        }
        return ( EXIT_SUCCESS );
    }

    

//  this is only for one dimensional
    int positivityConstraintPrimalRplus ( M_Model M, M_Variable::t coeff ) const {
        unsigned int coneone_n;
        unsigned int conetwo_n;
        
        

        if ( n % 2 == 0 ) { //even
            coneone_n = n/2;
            conetwo_n = coneone_n-1;
        } else { //odd
            coneone_n = std::max<unsigned int> ( ( n-1 ) /2,0 );
            conetwo_n = coneone_n;
        }


        const RRCA::MultiIndexSet<iVector> myIndex ( d,n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > indexMap;
        const RRCA::MultiIndexSet<iVector> mySmallIndex ( d,coneone_n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap;

        const RRCA::MultiIndexSet<iVector> mySmallIndex_2 ( d,conetwo_n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap_2;



        //  this part is the same for all possibilities
        const auto &mySet = mySmallIndex.get_MultiIndexSet();
//      copy into vector for simplicity
        std::vector<iVector> inter;
        for ( const auto &ind1 : mySet ) {
            inter.push_back ( ind1 );
        }
        for ( int j = 0; j < inter.size(); ++j ) {
            for ( int i = 0; i <= j; ++i ) {
                iVector newIndex = inter[i] + inter[j];
                eigenIJVector ij;
                ij << i, j;
                auto it = smallIndexMap.find ( newIndex );
                if ( it != smallIndexMap.end() ) {
                    it->second.push_back ( ij );
                } else {
                    smallIndexMap[newIndex].push_back ( ij );
                }
            }
        }



        //  same for the second cone
        const auto &mySet_2 = mySmallIndex_2.get_MultiIndexSet();
//      copy into vector for simplicity
        std::vector<iVector> inter_2;
        for ( const auto &ind1 : mySet_2 ) {
            inter_2.push_back ( ind1 );
        }
        iVector addition = iVector::Constant ( 1,1 );
        for ( int j = 0; j < inter_2.size(); ++j ) {
            for ( int i = 0; i <= j; ++i ) {
                iVector newIndex = inter_2[i] + inter_2[j]+addition;
                eigenIJVector ij;
                ij << i, j;
                auto it = smallIndexMap_2.find ( newIndex );
                if ( it != smallIndexMap_2.end() ) {
                    it->second.push_back ( ij );
                } else {
                    smallIndexMap_2[newIndex].push_back ( ij );
                }
            }
        }

        //      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
        M_Variable::t MM = M->variable ("SemiCone1", M_Domain::inPSDCone ( ( int ) coneone_n +1) ); // the moment matrix
        M_Variable::t MM_2 = M->variable ("SemiCone2", M_Domain::inPSDCone ( ( int ) conetwo_n +1) ); // the moment matrix
//         the coefficients



//      find the coefficients of the big coefficient vector in the matrix of the quadratic form
        const auto &mySet2 = myIndex.get_MultiIndexSet();
        unsigned int counter = 0;
        for ( const auto &ind : mySet2 ) {
//          find it in the map
//          the first cone
            const std::vector<eigenIJVector> ind1 = smallIndexMap[ind];
            M_Expression::t ee = M_Expr::constTerm ( 0.0 );
            for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
                if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
                    ee = M_Expr::add ( ee,M_Expr::mul ( 2.0,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) ) );
                } else {// diagonal
                    ee = M_Expr::add ( ee,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) );
                }
            }
//          and the second
            if ( counter>0 ) { // because the constant will never be in this set
                const std::vector<eigenIJVector> ind2 = smallIndexMap_2[ind];
                for ( Eigen::Index k = 0; k < ind2.size(); ++k ) {
                    if ( ind2[k] ( 0 ) != ind2[k] ( 1 ) ) {
                        ee = M_Expr::add ( ee,M_Expr::mul ( 2.0,MM_2->index ( ind2[k] ( 0 ),ind2[k] ( 1 ) ) ) );
                    } else {// diagonal
                        ee = M_Expr::add ( ee,MM_2->index ( ind2[k] ( 0 ),ind2[k] ( 1 ) ) );
                    }
                }
            }
            M->constraint ( M_Expr::sub ( ee,coeff->index ( counter ) ),M_Domain::equalsTo ( 0.0 ) );
            ++counter;
        }
        return(EXIT_SUCCESS);
    }
    
    //  this is only for one dimensional supported on interval [a,b]
    int positivityConstraintPrimalInterval ( M_Model M, M_Variable::t coeff, double a, double b ) const {
        unsigned int coneone_n;
        unsigned int conetwo_n;
        
        
        const bool iseven(n % 2 == 0);

        if ( iseven) { //even
            coneone_n = n/2;
            conetwo_n = coneone_n-1;
        } else { //odd
            coneone_n = std::max<unsigned int> ( ( n-1 ) /2,0 );
            conetwo_n = coneone_n;
        }


        const RRCA::MultiIndexSet<iVector> myIndex ( d,n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > indexMap;
        const RRCA::MultiIndexSet<iVector> mySmallIndex ( d,coneone_n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap_1;
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap_2; // need a second one

        const RRCA::MultiIndexSet<iVector> mySmallIndex_2 ( d,conetwo_n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap_3;
        std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap_4;
        
//      we also need 4 index maps. If even then we need 



        if(iseven){
            const auto &mySet = mySmallIndex.get_MultiIndexSet();
//          copy into vector for simplicity
            std::vector<iVector> inter;
            for ( const auto &ind1 : mySet ) {
                inter.push_back ( ind1 );
            }
            for ( int j = 0; j < inter.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = inter[i] + inter[j];
                    eigenIJVector ij;
                    ij << i, j;
                    auto it = smallIndexMap_1.find ( newIndex );
                    if ( it != smallIndexMap_1.end() ) {
                        it->second.push_back ( ij );
                    } else {
                        smallIndexMap_1[newIndex].push_back ( ij );
                    }
                }
            }
//          now we need the remaining three maps for the smaller cone
            //  same for the second cone
            const auto &mySet_2 = mySmallIndex_2.get_MultiIndexSet();
//          copy into vector for simplicity
            std::vector<iVector> inter_2;
            for ( const auto &ind1 : mySet_2 ) {
                inter_2.push_back ( ind1 );
            }
//          we need to multiply by - ab+(a+b)t -t^2
//          start with the constant
            for ( int j = 0; j < inter_2.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = inter_2[i] + inter_2[j];
                    eigenIJVector ij;
                    ij << i, j;
                    auto it = smallIndexMap_2.find ( newIndex );
                    if ( it != smallIndexMap_2.end() ) {
                        it->second.push_back ( ij );
                    } else {
                        smallIndexMap_2[newIndex].push_back ( ij );
                    }
                }
            }
//          now the linear part
            iVector vec_1 = iVector::Constant ( 1,1 );
            for ( int j = 0; j < inter_2.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = inter_2[i] + inter_2[j]+vec_1;
                    eigenIJVector ij;
                    ij << i, j;
                    auto it = smallIndexMap_3.find ( newIndex );
                    if ( it != smallIndexMap_3.end() ) {
                        it->second.push_back ( ij );
                    } else {
                        smallIndexMap_3[newIndex].push_back ( ij );
                    }
                }
            }
            //          now the quadratic part
            iVector vec_2 = iVector::Constant ( 1,2 );
            for ( int j = 0; j < inter_2.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = inter_2[i] + inter_2[j]+vec_2;
                    eigenIJVector ij;
                    ij << i, j;
                    auto it = smallIndexMap_4.find ( newIndex );
                    if ( it != smallIndexMap_4.end() ) {
                        it->second.push_back ( ij );
                    } else {
                        smallIndexMap_4[newIndex].push_back ( ij );
                    }
                }
            }
        } else { // n is odd
//          we have two maps for the first cone and two for the second
//          first cone constant and linear
            iVector vec_1 = iVector::Constant ( 1,1 );
//             iVector vec_2 = iVector::Constant ( 1,2 );
            const auto &mySet = mySmallIndex.get_MultiIndexSet();
//          copy into vector for simplicity
            std::vector<iVector> inter;
            for ( const auto &ind1 : mySet ) {
                inter.push_back ( ind1 );
            }
//          constant part 
            for ( int j = 0; j < inter.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = inter[i] + inter[j];
                    eigenIJVector ij;
                    ij << i, j;
                    auto it = smallIndexMap_1.find ( newIndex );
                    if ( it != smallIndexMap_1.end() ) {
                        it->second.push_back ( ij );
                    } else {
                        smallIndexMap_1[newIndex].push_back ( ij );
                    }
                }
            }
            //          linear part 
            for ( int j = 0; j < inter.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = inter[i] + inter[j]+vec_1;
                    eigenIJVector ij;
                    ij << i, j;
                    auto it = smallIndexMap_2.find ( newIndex );
                    if ( it != smallIndexMap_2.end() ) {
                        it->second.push_back ( ij );
                    } else {
                        smallIndexMap_2[newIndex].push_back ( ij );
                    }
                }
            }
            
//          second cone
            const auto &mySet_2 = mySmallIndex_2.get_MultiIndexSet();
//          copy into vector for simplicity
            std::vector<iVector> inter_2;
            for ( const auto &ind1 : mySet_2 ) {
                inter_2.push_back ( ind1 );
            }
//          constant part 
            for ( int j = 0; j < inter_2.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = inter_2[i] + inter_2[j];
                    eigenIJVector ij;
                    ij << i, j;
                    auto it = smallIndexMap_3.find ( newIndex );
                    if ( it != smallIndexMap_3.end() ) {
                        it->second.push_back ( ij );
                    } else {
                        smallIndexMap_3[newIndex].push_back ( ij );
                    }
                }
            }
            //          linear part 
            for ( int j = 0; j < inter_2.size(); ++j ) {
                for ( int i = 0; i <= j; ++i ) {
                    iVector newIndex = inter_2[i] + inter_2[j]+vec_1;
                    eigenIJVector ij;
                    ij << i, j;
                    auto it = smallIndexMap_4.find ( newIndex );
                    if ( it != smallIndexMap_4.end() ) {
                        it->second.push_back ( ij );
                    } else {
                        smallIndexMap_4[newIndex].push_back ( ij );
                    }
                }
            }
        }


        //      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
        M_Variable::t MM = M->variable ("SemiCone1", M_Domain::inPSDCone ( ( int ) coneone_n +1) ); // the moment matrix
        M_Variable::t MM_2 = M->variable ("SemiCone2", M_Domain::inPSDCone ( ( int ) conetwo_n +1) ); // the moment matrix
//         the coefficients



//      find the coefficients of the big coefficient vector in the matrix of the quadratic form
        const auto &mySet_3 = myIndex.get_MultiIndexSet();
        unsigned int counter = 0;
        if(iseven){
            for ( const auto &ind : mySet_3 ) {
//          find it in the map
//          constant parts
//                 first cone
                const std::vector<eigenIJVector> ind1 = smallIndexMap_1[ind];
                M_Expression::t ee = M_Expr::constTerm ( 0.0 );
                for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
                    if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
                        ee = M_Expr::add ( ee,M_Expr::mul ( 2.0,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) ) );
                    } else {// diagonal
                        ee = M_Expr::add ( ee,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) );
                    }
                }
//                 second cone
                const std::vector<eigenIJVector> ind2 = smallIndexMap_2[ind];
                for ( Eigen::Index k = 0; k < ind2.size(); ++k ) {
                    if ( ind2[k] ( 0 ) != ind2[k] ( 1 ) ) {
                        ee = M_Expr::add ( ee,M_Expr::mul ( -b*a*2.0,MM_2->index ( ind2[k] ( 0 ),ind2[k] ( 1 ) ) ) );
                    } else {// diagonal
                        ee = M_Expr::add ( ee,M_Expr::mul (-b*a, MM_2->index ( ind2[k] ( 0 ),ind2[k] ( 1 ) ) ));
                    }
                }   
                
//          and the linear part
                if ( counter>0 ) { // because the constant will never be in this set
                    const std::vector<eigenIJVector> ind3 = smallIndexMap_3[ind];
                    for ( Eigen::Index k = 0; k < ind3.size(); ++k ) {
                        if ( ind3[k] ( 0 ) != ind3[k] ( 1 ) ) {
                            ee = M_Expr::add ( ee,M_Expr::mul ( (a+b)*2.0,MM_2->index ( ind3[k] ( 0 ),ind3[k] ( 1 ) ) ) );
                        } else {// diagonal
                            ee = M_Expr::add ( ee,M_Expr::mul ( (a+b),MM_2->index ( ind3[k] ( 0 ),ind3[k] ( 1 ) ) ));
                        }
                    }
                }
                if( counter > 1){ // quadratic part
                    const std::vector<eigenIJVector> ind4 = smallIndexMap_4[ind];
                    for ( Eigen::Index k = 0; k < ind4.size(); ++k ) {
                        if ( ind4[k] ( 0 ) != ind4[k] ( 1 ) ) {
                            ee = M_Expr::add ( ee,M_Expr::mul ( -2.0,MM_2->index ( ind4[k] ( 0 ),ind4[k] ( 1 ) ) ) );
                        } else {// diagonal
                            ee = M_Expr::add ( ee,M_Expr::mul ( -1.0,MM_2->index ( ind4[k] ( 0 ),ind4[k] ( 1 ) ) ) );
                        }
                    }
                }
                M->constraint ( M_Expr::sub ( ee,coeff->index ( counter ) ),M_Domain::equalsTo ( 0.0 ) );
                ++counter;
            }
        } else { // odd
            for ( const auto &ind : mySet_3 ) {
//          find it in the map
//          constant parts
                const std::vector<eigenIJVector> ind1 = smallIndexMap_1[ind];
                M_Expression::t ee = M_Expr::constTerm ( 0.0 );
                for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
                    if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
                        ee = M_Expr::add ( ee,M_Expr::mul ( 2.0*b,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ) ) );
                    } else {// diagonal
                        ee = M_Expr::add ( ee,M_Expr::mul (b,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) )) );
                    }
                }
                const std::vector<eigenIJVector> ind2 = smallIndexMap_3[ind];
                for ( Eigen::Index k = 0; k < ind2.size(); ++k ) {
                    if ( ind2[k] ( 0 ) != ind2[k] ( 1 ) ) {
                        ee = M_Expr::add ( ee,M_Expr::mul ( -a*2.0,MM_2->index ( ind2[k] ( 0 ),ind2[k] ( 1 ) ) ) );
                    } else {// diagonal
                        ee = M_Expr::add ( ee,M_Expr::mul (-a, MM_2->index ( ind2[k] ( 0 ),ind2[k] ( 1 ) ) ) );
                    }
                }   
                
//          and the linear parts
                if ( counter>0 ) { // because the constant will never be in this set
                    const std::vector<eigenIJVector> ind3 = smallIndexMap_2[ind];
                    for ( Eigen::Index k = 0; k < ind3.size(); ++k ) {
                        if ( ind3[k] ( 0 ) != ind3[k] ( 1 ) ) {
                            ee = M_Expr::add ( ee,M_Expr::mul ( -2.0,MM->index ( ind3[k] ( 0 ),ind3[k] ( 1 ) ) ) );
                        } else {// diagonal
                            ee = M_Expr::add ( ee,M_Expr::mul (-1.0, MM->index ( ind3[k] ( 0 ),ind3[k] ( 1 ) ) )  );
                        }
                    }
                    const std::vector<eigenIJVector> ind4 = smallIndexMap_4[ind];
                    for ( Eigen::Index k = 0; k < ind4.size(); ++k ) {
                        if ( ind4[k] ( 0 ) != ind4[k] ( 1 ) ) {
                            ee = M_Expr::add ( ee,M_Expr::mul ( 2.0,MM_2->index ( ind4[k] ( 0 ),ind4[k] ( 1 ) ) ) );
                        } else {// diagonal
                            ee = M_Expr::add ( ee,MM_2->index ( ind4[k] ( 0 ),ind4[k] ( 1 ) ) );
                        }
                    }
                }
                M->constraint ( M_Expr::sub ( ee,coeff->index ( counter ) ),M_Domain::equalsTo ( 0.0 ) );
                ++counter;
            }
        }
        
        return(EXIT_SUCCESS);
    }

    

    int  fillEqualityConstraints ( M_Model M,
                                   M_Variable::t coeff,
                                   const Matrix& H,
                                   const Matrix& S,
                                   const Vector& rhsS ) const {


        assert ( S.rows() == rhsS.size() );
        assert ( S.cols() == H.rows() );
        assert( H.rows() == coeff->getSize());
        
        


//         now the constraints
        Matrix equVec = S * H;
//      pointless copy bc monty can not deal with const pointers
        Vector rhss = rhsS;
//      intermediate wrapper for constness

        const M_Matrix::t equVecWrap_t = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new  M_ndarray_2 ( equVec.data(), monty::shape ( equVec.cols(),equVec.rows() ) ) ) );

        const auto equVecRhsWrap = std::shared_ptr<M_ndarray_1 > ( new   M_ndarray_1 ( rhss.data(),  monty::shape ( rhss.size() ) ) );
        M->constraint ("equality", M_Expr::sub ( M_Expr::mul ( equVecWrap_t->transpose(),coeff ),equVecRhsWrap ), M_Domain::equalsTo ( 0.0 ) );
        return ( EXIT_SUCCESS );
    }

    int  fillInequalityConstraints ( M_Model M,
                                     M_Variable::t coeff,
                                     const Matrix& H,
                                     const Matrix& U,
                                     const Vector& rhsU ) const {

        assert ( U.rows() == rhsU.size() );
        assert ( U.cols() == H.rows() );
        
        


//         now the constraints
        Matrix inequVec = U * H;
//      pointless copy bc monty can not deal with const pointers
        Vector rhss = rhsU;
//      intermediate wrapper for constness

        const M_Matrix::t inequVecWrap_t = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new  M_ndarray_2 ( inequVec.data(), monty::shape ( inequVec.cols(),inequVec.rows() ) ) ) );

        const auto inequVecRhsWrap = std::shared_ptr<M_ndarray_1 > ( new   M_ndarray_1 ( rhss.data(),  monty::shape ( rhss.size() ) ) );
        M->constraint ( "inequality", M_Expr::sub ( M_Expr::mul ( inequVecWrap_t->transpose(),coeff ),inequVecRhsWrap ), M_Domain::lessThan ( 0.0 ) );
        return ( EXIT_SUCCESS );
    }

    int  fillPrimalObjective ( M_Model M,
                               M_Variable::t coeff,
                               const Matrix& H ) const {

        
        

        M_Variable::t u = M->variable ( "obj", 1, M_Domain::greaterThan ( 0.0 ) );
        Eigen::SelfAdjointEigenSolver<Matrix> eig0(H);
        Matrix Hsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
//         std::cout << "eigenvalues primal " << eig0.eigenvalues()  << std::endl;
        
        
        M_Matrix::t UUwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( Hsqrt.data(), monty::shape ( Hsqrt.rows(), Hsqrt.cols() ) ) ) );
        M->constraint ( M_Expr::vstack ( 0.5,u, M_Expr::mul ( UUwrap,coeff  ) ), M_Domain::inRotatedQCone() );
        M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::mul (0.5,u ));
        return ( EXIT_SUCCESS );
    }

   


};


/*

// base class for embedding positive polynomials
class PolynomialEmbedding
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> iVector;
    typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> eigenIndexMatrix;
    typedef Eigen::Matrix<Eigen::Index, 2, 1> eigenIJVector;


public:
    PolynomialEmbedding ( unsigned int d_, unsigned int n_, const Matrix& H_ ) : d ( d_ ),
    n ( n_ ),
    initN ( binomialCoefficient( n_+d_, d_ ) ),
    smallN ( binomialCoefficient( n_/2+d_, d_ ) ),
    myIndex ( d_,n_ ),
    mySmallIndex ( d_,n_/2 ),
    H ( H_ ){
        assert ( H.rows() == initN );
        assert ( H.cols() == initN );
        assert ( n_ % 2 == 0 );

//         this is for the positivity
        const auto &mySet = mySmallIndex.get_MultiIndexSet();
//      copy into vector for simplicity
        std::vector<iVector> inter;
        for ( const auto &ind1 : mySet ) {
            inter.push_back ( ind1 );
        }
        for ( int j = 0; j < inter.size(); ++j ) {
            for ( int i = 0; i <= j; ++i ) {
                iVector newIndex = inter[i] + inter[j];
                eigenIJVector ij;
                ij << i, j;
                auto it = smallIndexMap.find ( newIndex );
                if ( it != smallIndexMap.end() )
                    it->second.push_back ( ij );
                else
                    smallIndexMap[newIndex].push_back ( ij );
            }
        }
    }

//  return mosek model object that contains the positivity conditions and the mininum norm objective
    M_Model getModel(){
         
        

        M = new mosek::fusion::Model ( "PositivePolynomial" );
        M->setLogHandler ( [=] ( const std::string & msg ) {
            std::cout << msg << std::flush;
        } );
//         auto _M = finally ( [&]() {
//             M->dispose();
//         } );
//      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
        M_Variable::t MM = M->variable ( M_Domain::inPSDCone ( ( int ) smallN ) ); // the moment matrix
//         the coefficients
        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded());
        M_Variable::t u = M->variable ( "obj", 1, M_Domain::greaterThan(0.0));



//      find the coefficients of the big coefficient vector in the matrix of the quadratic form
        const auto &mySet = myIndex.get_MultiIndexSet();
        unsigned int counter = 0;
        for ( const auto &ind : mySet ) {
//          find it in the map
            const std::vector<eigenIJVector> ind1 = smallIndexMap[ind];
            M_Expression::t ee = M_Expr::constTerm(0.0);
            for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
                if(ind1[k] ( 0 ) != ind1[k] ( 1 )){
                    ee = M_Expr::add(ee,M_Expr::mul(2.0,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) )));
                } else {// diagonal
                    ee = M_Expr::add(ee,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ));
                }
            }
            M->constraint ( M_Expr::sub ( ee,h->index(counter)),M_Domain::equalsTo ( 0.0 ) );
            ++counter;
        }
//      compute Cholesky decopm of H
        Matrix UU = H.llt().matrixU(); // upper triangular so we can give it to mosek in row major as L
        M_Matrix::t UUwrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( UU.data(), monty::shape ( initN, initN ) ) ));
        M->constraint(M_Expr::vstack(0.5,u, M_Expr::mul(h , UUwrap)), M_Domain::inRotatedQCone());
        M->objective ( mosek::fusion::ObjectiveSense::Minimize, u);

        return(M);
    }


    //  return mosek model object that contains the positivity conditions and the mininum norm objective
//     the matrix S contains the equality constraints, and matrix U the inequality constraints. rhs are the corresponding right-hand sides
    M_Model getConstrainedModel(const Matrix& S, Vector& rhsS, const Matrix& U,  Vector& rhsU){
         
        

        assert ( S.rows() == rhsS.size() );
        assert ( U.rows() == rhsU.size() );
        assert ( U.cols() == H.rows() );
        assert ( S.cols() == H.rows() );

        M = new mosek::fusion::Model ( "ConstrainedPositivePolynomial" );
        M->setLogHandler ( [=] ( const std::string & msg ) {
            std::cout << msg << std::flush;
        } );
//         auto _M = finally ( [&]() {
//             M->dispose();
//         } );
//      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
        M_Variable::t MM = M->variable ( M_Domain::inPSDCone ( ( int ) smallN ) ); // the moment matrix
//         the coefficients
        M_Variable::t h = M->variable ( "coefficients", initN, M_Domain::unbounded());
        M_Variable::t u = M->variable ( "obj", 1, M_Domain::greaterThan(0.0));



//      find the coefficients of the big coefficient vector in the matrix of the quadratic form
        const auto &mySet = myIndex.get_MultiIndexSet();
        unsigned int counter = 0;
        for ( const auto &ind : mySet ) {
//          find it in the map
            const std::vector<eigenIJVector> ind1 = smallIndexMap[ind];
            M_Expression::t ee = M_Expr::constTerm(0.0);
            for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
                if(ind1[k] ( 0 ) != ind1[k] ( 1 )){
                    ee = M_Expr::add(ee,M_Expr::mul(2.0,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) )));
                } else {// diagonal
                    ee = M_Expr::add(ee,MM->index ( ind1[k] ( 0 ),ind1[k] ( 1 ) ));
                }
            }
            M->constraint ( M_Expr::sub ( ee,h->index(counter)),M_Domain::equalsTo ( 0.0 ) );
            ++counter;
        }
//      compute Cholesky decopm of H
        Matrix UU = H.llt().matrixU(); // upper triangular so we can give it to mosek in row major as L
        M_Matrix::t UUwrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( UU.data(), monty::shape ( initN, initN ) ) ));
        M->constraint(M_Expr::vstack(0.5,u, M_Expr::mul(h , UUwrap)), M_Domain::inRotatedQCone());


//         now the constraints
        Matrix equVec = S * H;
        Matrix inequVec = U * H;
        M_Matrix::t equVecWrap_t = M_Matrix::dense( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( equVec.data(), monty::shape ( equVec.cols(),equVec.rows() ) ) ));
        M_Matrix::t inequVecWrap_t = M_Matrix::dense( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( inequVec.data(), monty::shape ( inequVec.cols(),inequVec.rows() ) ) ));



        const auto equVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhsS.data(),  monty::shape ( rhsS.size()) ) );
        const auto inequVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhsU.data(),  monty::shape ( rhsU.size()) ) );
//         const auto inequVecRhsWrap = std::shared_ptr<M_ndarray_1 > ( new M_ndarray_1 ( rhsU.data(), monty::shape ( rhsU.size() ) ) );

        M->constraint(M_Expr::sub(M_Expr::mul(equVecWrap_t->transpose(),h),equVecRhsWrap), M_Domain::equalsTo(0.0));
        M->constraint(M_Expr::sub(M_Expr::mul(inequVecWrap_t->transpose(),h),inequVecRhsWrap), M_Domain::lessThan(0.0));



        M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::mul(0.5,u));

        return(M);
    }


        //  return mosek model object that contains the positivity conditions and the mininum norm objective
//     the matrix S contains the equality constraints, and matrix U the inequality constraints. rhs are the corresponding right-hand sides
    M_Model getConstrainedDualModel(const Matrix& S, Vector& rhsS, const Matrix& U,  Vector& rhsU){
         
        

        assert ( S.rows() == rhsS.size() );
        assert ( U.rows() == rhsU.size() );
        assert ( U.cols() == H.rows() );
        assert ( S.cols() == H.rows() );

        const int Urow(U.rows());
        const int Srow(S.rows());

        M = new mosek::fusion::Model ( "ConstrainedPositivePolynomialDual" );
        M->setLogHandler ( [=] ( const std::string & msg ) {
            std::cout << msg << std::flush;
        } );
//         auto _M = finally ( [&]() {
//             M->dispose();
//         } );
//      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
        M_Variable::t MM = M->variable ("mom", M_Domain::inPSDCone ( ( int ) smallN ) ); // the moment matrix
//         the coefficients
        M_Variable::t nu = M->variable ( "nu", initN, M_Domain::unbounded());
        M_Variable::t epsilon = M->variable ( "epsilon", Urow, M_Domain::greaterThan(0.0));
        M_Variable::t eta = M->variable ( "eta", Srow, M_Domain::unbounded());
        M_Variable::t u = M->variable ( "obj", 1, M_Domain::greaterThan(0.0));
        M_Expression::t theta = M_Expr::vstack(eta,nu,epsilon);



//      the constraints for the positive semidefinite condition
        const auto &mySet = myIndex.get_MultiIndexSet();
        unsigned int counter = 0;
//         const M_Matrix::t L = M_Matrix::dense(( int ) smallN, ( int ) smallN, 0.0);



        M_Expression::t bige = M_Expr::zeros(nint({( int ) smallN,( int ) smallN}));
        for ( const auto &ind : mySet ) {
//          find it in the map
            const std::vector<eigenIJVector> ind1 = smallIndexMap[ind];
//             M_Expression::t ee = M_Expr::constTerm(L);
            Matrix oida(smallN, smallN);
            std::cout << " nu " << counter << std::endl;
            oida.setZero();
            for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
//                 bige->index ( nint({ind1[k] ( 0 ),ind1[k] ( 1 ) })) = M_Expr::add(bige->index ( nint({ind1[k] ( 0 ),ind1[k] ( 1 ) })),nu->index(counter));
                oida(ind1[k] ( 0 ),ind1[k] ( 1 ))+=  1;
//                 M->constraint(M_Expr::sub(MM->index(ind1[k] ( 0 ),ind1[k] ( 1 )),nu->index(counter)),M_Domain::equalsTo ( 0.0 ) );
                if(ind1[k] ( 0 ) != ind1[k] ( 1 )){
//                     bige->index ( nint({ind1[k] ( 1 ),ind1[k] ( 0 ) })) = M_Expr::add(bige->index (nint({ind1[k] ( 1 ),ind1[k] ( 0 ) })),nu->index(counter));
                    oida(ind1[k] ( 1 ),ind1[k] ( 0 ))+=  1;
                }
            }
            M_Matrix::t Lwrap = M_Matrix::dense(std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( oida.data(), monty::shape (( int ) smallN,( int ) smallN ) ) ));
            bige = M_Expr::add(bige,M_Expr::mul(nu->index(counter),Lwrap));
            std::cout << oida << std::endl;
            ++counter;
        }
//         M->constraint(M_Expr::sub(MM->index(0,0),nu->index(0)),M_Domain::equalsTo ( 0.0 ) );
//         M->constraint(M_Expr::sub(MM->index(1,0),nu->index(1)),M_Domain::equalsTo ( 0.0 ) );
//         M->constraint(M_Expr::sub(MM->index(2,0),nu->index(2)),M_Domain::equalsTo ( 0.0 ) );
//         M->constraint(M_Expr::sub(MM->index(1,1),nu->index(2)),M_Domain::equalsTo ( 0.0 ) );
//         M->constraint(M_Expr::sub(MM->index(2,1),nu->index(3)),M_Domain::equalsTo ( 0.0 ) );
//         M->constraint(M_Expr::sub(MM->index(2,2),nu->index(4)),M_Domain::equalsTo ( 0.0 ) );

        M->constraint ( M_Expr::sub ( bige,MM),M_Domain::equalsTo ( 0.0 ) );
//         for(int i = 0; i < smallN; ++i){
//             for(int j = 0; j < smallN; ++j){
//                 std::cout << bige->index(nint({i,j }))->toString() << '\t';
//             }
//             std::cout << std::endl;
//         }


        M->writeTask("dump.task.gz");

//      now construct the big matrix for the quadratic form


        //         now the constraints
        Matrix equVec = S * H;
        Matrix inequVec = U * H;

        Matrix bigMat(Srow + H.rows() + Urow, Srow+H.rows()+Urow);

//         std::cout << "bigmat dim " << eigenDim(bigMat) << std::endl;
//         std::cout  << eigenDim(equVec * S.transpose()) << '\t' << eigenDim(S) << '\t' << eigenDim(- equVec* U.transpose()) << std::endl;
//         std::cout  << eigenDim(S.transpose()) << '\t' << eigenDim(H.llt().solve(Matrix::Identity(H.rows(), H.rows()))) << '\t' << eigenDim(-U.transpose()) << std::endl;
//         std::cout  << eigenDim(-inequVec * S.transpose()) << '\t' << eigenDim(-U) << '\t' << eigenDim(U * H * U.transpose()) << std::endl;
//
//
//
        bigMat << equVec * S.transpose() ,              S,                                                          - equVec* U.transpose(),
                 S.transpose(),                         H.llt().solve(Matrix::Identity(H.rows(), H.rows())),   -U.transpose(),
                 -inequVec * S.transpose(),             -U,                                                         inequVec * U.transpose();

// /*
        M_Matrix::t equVecWrap_t = M_Matrix::dense( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( equVec.data(), monty::shape ( equVec.cols(),equVec.rows() ) ) ));
        M_Matrix::t inequVecWrap_t = M_Matrix::dense( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( inequVec.data(), monty::shape ( inequVec.cols(),inequVec.rows() ) ) ));*/

// const auto equVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhsS.data(),  monty::shape ( rhsS.size() ) ) );
// const auto inequVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhsU.data(),  monty::shape ( rhsU.size() ) ) );
// //         const auto inequVecRhsWrap = std::shared_ptr<M_ndarray_1 > ( new M_ndarray_1 ( rhsU.data(), monty::shape ( rhsU.size() ) ) );
// //      compute Cholesky decopm of H
// Matrix UU = bigMat.llt().matrixU(); // upper triangular so we can give it to mosek in row major as L
// M_Matrix::t UUwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( UU.data(), monty::shape ( bigMat.rows(), bigMat.cols() ) ) ) );
// M->constraint ( M_Expr::vstack ( 0.5,u, M_Expr::mul ( theta, UUwrap ) ), M_Domain::inRotatedQCone() );
// 
// 
// M->objective ( ObjectiveSense::Maximize, M_Expr::add ( M_Expr::mul ( -0.5,u ),M_Expr::sub ( M_Expr::dot ( eta,equVecRhsWrap ),M_Expr::dot ( epsilon,inequVecRhsWrap ) ) ) );
// 
// return ( M );
// }
// 
// 
// 
// private:
// const  unsigned int d;
// const  unsigned int n;
// const  unsigned int initN;
// const  unsigned int smallN;
// const RRCA::MultiIndexSet<iVector> myIndex; // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
// std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > indexMap;
// const RRCA::MultiIndexSet<iVector> mySmallIndex; // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
// std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap;
// 
// const Matrix& H;
// 
// M_Model M;
// 
// 
// };
// */



// M_Model solveEqualityConstrainedDual ( const Matrix& S, const Vector& rhsS ) {
//         
//         
// 
//         M_Model M = new mosek::fusion::Model ( "EqualityConstrainedPolynomialEmbeddingDual" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
// 
//         M_Variable::t nu = M->variable ( "nu", initN, M_Domain::unbounded());
//         M_Variable::t eta = M->variable ( "eta", S.rows(), M_Domain::unbounded());
//         
//         int success;
//         if ( sp == StateSpace::Rplus && d ==1 ) {
//             success = positivityConstraintDualRplus ( M, nu);
//         } else {
//             success = positivityConstraintDualRn ( M, nu);
//         }
// 
// 
//         success = fillEqualityConstrainedDualObjective(M,nu,eta,H,S,rhsS);
// 
//         return ( M );
//     }
// 
//     M_Model solveInequalityConstrainedDual ( const Matrix& U, const Vector& rhsU ) {
//                 
//         
// 
//         M_Model M = new mosek::fusion::Model ( "InequalityConstrainedPolynomialEmbeddingDual" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
// 
//         M_Variable::t nu = M->variable ( "nu", initN, M_Domain::unbounded());
//         M_Variable::t epsilon = M->variable ( "epsilon", U.rows(), M_Domain::greaterThan(0.0));
//         
//         int success;
//         if ( sp == StateSpace::Rplus && d ==1 ) {
//             success = positivityConstraintDualRplus ( M, nu );
//         } else {
//             success = positivityConstraintDualRn ( M, nu );
//         }
// 
// 
//         success = fillInequalityConstrainedDualObjective(M,nu,epsilon,H,U,rhsU);
// 
//         return ( M );
//     }
// 
//     M_Model solveConstrainedDual ( const Matrix& S, 
//                                                    const Vector& rhsS, 
//                                                    const Matrix& U, 
//                                                    const Vector& rhsU ) {
//         
//         
// 
//         M_Model M = new mosek::fusion::Model ( "ConstrainedPolynomialEmbeddingDual" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
// 
//         M_Variable::t nu = M->variable ( "nu", initN, M_Domain::unbounded());
//         M_Variable::t epsilon = M->variable ( "epsilon", U.rows(), M_Domain::greaterThan(0.0));
//         M_Variable::t eta = M->variable ( "eta", S.rows(), M_Domain::unbounded());
//         
//        
//         int success;
//         if ( sp == StateSpace::Rplus && d ==1 ) {
//             success = positivityConstraintDualRplus ( M, nu);
//         } else {
//             success = positivityConstraintDualRn ( M, nu );
//         }
// 
// 
//         success = fillConstrainedDualObjective(M,nu,eta,epsilon,H,S,rhsS,U,rhsU);
// 
//         return ( M );
//     }
//     
//         int positivityConstraintDualRn ( M_Model M, M_Variable::t nu ) const {
//         assert ( n % 2 == 0 );
//         
//         
//         const  unsigned int smallN ( binomialCoefficient ( n/2+d, d ) );
//         const RRCA::MultiIndexSet<iVector> myIndex ( d,n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
//         std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > indexMap;
//         const RRCA::MultiIndexSet<iVector> mySmallIndex ( d,n/2 ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
//         std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap;
// 
//         //  this part is the same for all possibilities
//         const auto &mySet = mySmallIndex.get_MultiIndexSet();
// //      copy into vector for simplicity
//         std::vector<iVector> inter;
//         for ( const auto &ind1 : mySet ) {
//             inter.push_back ( ind1 );
//         }
//         for ( int j = 0; j < inter.size(); ++j ) {
//             for ( int i = 0; i <= j; ++i ) {
//                 iVector newIndex = inter[i] + inter[j];
//                 eigenIJVector ij;
//                 ij << i, j;
//                 auto it = smallIndexMap.find ( newIndex );
//                 if ( it != smallIndexMap.end() ) {
//                     it->second.push_back ( ij );
//                 } else {
//                     smallIndexMap[newIndex].push_back ( ij );
//                 }
//             }
//         }
// 
//         //      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
//         M_Variable::t MM = M->variable ("SemiCone1", M_Domain::inPSDCone ( ( int ) smallN ) ); // the moment matrix
// //         the coefficients
// 
// 
//         int counter ( 0 );
//         M_Expression::t bige = M_Expr::zeros ( nint ( { ( int ) smallN, ( int ) smallN} ) );
//         const auto &mySet2 = myIndex.get_MultiIndexSet();
//         for ( const auto &ind : mySet2 ) {
// //          find it in the map
//             const std::vector<eigenIJVector> ind1 = smallIndexMap[ind];
//             Matrix oida ( smallN, smallN );
//             oida.setZero();
//             for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
//                 oida ( ind1[k] ( 0 ),ind1[k] ( 1 ) )+=  1;
//                 if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
//                     oida ( ind1[k] ( 1 ),ind1[k] ( 0 ) )+=  1;
//                 }
//             }
// //             std::cout << " j " << counter << " L " << std::endl << oida << std::endl;
//             M_Matrix::t Lwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( oida.data(), monty::shape ( ( int ) smallN, ( int ) smallN ) ) ) );
//             bige = M_Expr::add ( bige,M_Expr::mul ( nu->index ( counter ),Lwrap ) );
//             ++counter;
//         }
//         
// //      scale the Matrix
// //         Eigen::SelfAdjointEigenSolver<Matrix> eig0(H);
// //         const double meanentry((eig0.eigenvectors() * eig0.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose()).mean());
//         
// //         M->constraint ( M_Expr::sub ( MM,M_Expr::mul(meanentry,bige)),M_Domain::equalsTo ( 0.0 ) );
//         M->constraint ( M_Expr::sub ( MM,bige),M_Domain::equalsTo ( 0.0 ) );
//         return ( EXIT_SUCCESS );
//     }
//     
//     //  this is only for one dimensional
//     int positivityConstraintDualRplus ( M_Model M, M_Variable::t nu) const {
//         
//         
// //         std::cout << "hier ! " << std::endl;
//         
//         unsigned int coneone_n;
//         unsigned int conetwo_n;
//         
//         
//         
// 
//         if ( n % 2 == 0 ) { //even
//             coneone_n = n/2;
//             conetwo_n = coneone_n-1;
//         } else { //odd
//             coneone_n = std::max<unsigned int> ( ( n-1 ) /2,0 );
//             conetwo_n = coneone_n;
//         }
// 
// 
//         const RRCA::MultiIndexSet<iVector> myIndex ( d,n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
//         std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > indexMap;
//         const RRCA::MultiIndexSet<iVector> mySmallIndex ( d,coneone_n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
//         std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap;
// 
//         const RRCA::MultiIndexSet<iVector> mySmallIndex_2 ( d,conetwo_n ); // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
//         std::map<iVector, std::vector<eigenIJVector>, RRCA::FMCA_Compare<iVector> > smallIndexMap_2;
// 
// 
// 
//         //  this part is the same for all possibilities
//         const auto &mySet = mySmallIndex.get_MultiIndexSet();
// //      copy into vector for simplicity
//         std::vector<iVector> inter;
//         for ( const auto &ind1 : mySet ) {
//             inter.push_back ( ind1 );
// //             std::cout << "small order " << ind1 << std::endl;
//         }
//         for ( int j = 0; j < inter.size(); ++j ) {
//             for ( int i = 0; i <= j; ++i ) {
//                 iVector newIndex = inter[i] + inter[j];
//                 eigenIJVector ij;
//                 ij << i, j;
//                 auto it = smallIndexMap.find ( newIndex );
//                 if ( it != smallIndexMap.end() ) {
//                     it->second.push_back ( ij );
//                 } else {
//                     smallIndexMap[newIndex].push_back ( ij );
//                 }
//             }
//         }
//         
//         ////////////////
//         //  print map //
//         ////////////////
// //         std::cout << "small map " << std::endl;
// //         for (const auto &ind1 : mySet) std::cout << ind1.transpose() << std::endl;
// //         std::cout << "--------" << std::endl;
// //         for (const auto &ind1 : smallIndexMap) {
// //             std::cout << ind1.first.transpose() << std::endl;
// //                 for (const auto &ij : ind1.second) std::cout << ij.transpose() << "|";
// //                     std::cout << "\n--------" << std::endl;
// //         }
// 
// 
// 
//         //  same for the second cone
//         const auto &mySet_2 = mySmallIndex_2.get_MultiIndexSet();
// //      copy into vector for simplicity
//         std::vector<iVector> inter_2;
//         for ( const auto &ind1 : mySet_2 ) {
//             inter_2.push_back ( ind1 );
// //             std::cout << "small small index " << ind1 << std::endl;
//         }
//         iVector addition = iVector::Constant ( 1,1 );
//         for ( int j = 0; j < inter_2.size(); ++j ) {
//             for ( int i = 0; i <= j; ++i ) {
//                 iVector newIndex = inter_2[i] + inter_2[j]+addition;
// //                 std::cout << "new index " << newIndex << std::endl;
//                 eigenIJVector ij;
//                 ij << i, j;
//                 auto it = smallIndexMap_2.find ( newIndex );
//                 if ( it != smallIndexMap_2.end() ) {
//                     it->second.push_back ( ij );
//                 } else {
//                     smallIndexMap_2[newIndex].push_back ( ij );
//                 }
//             }
//         }
//         
//                 ////////////////
//         //  print map //
//         ////////////////
// //         std::cout << "small small map " << std::endl;
// //         for (const auto &ind1 : mySet_2) std::cout << ind1.transpose() << std::endl;
// //         std::cout << "--------" << std::endl;
// //         for (const auto &ind1 : smallIndexMap_2) {
// //             std::cout << ind1.first.transpose() << std::endl;
// //                 for (const auto &ij : ind1.second) std::cout << ij.transpose() << "|";
// //                     std::cout << "\n--------" << std::endl;
// //         }
// 
// 
// 
//         //      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
//         M_Variable::t MM = M->variable ( "SemiCone1", M_Domain::inPSDCone ( ( int ) (coneone_n+1 ) )); // the moment matrix
// //         the coefficients
//         const auto &mySet3 = myIndex.get_MultiIndexSet();
// 
// 
// //      find the coefficients of the big coefficient vector in the matrix of the quadratic form
//         unsigned int counter = 0;
//         M_Expression::t bige = M_Expr::zeros ( nint ( { ( int ) (coneone_n+1 ), ( int ) (coneone_n+1 )} ) );
//         for ( const auto &ind : mySet3 ) {
// //          find it in the map
//             const std::vector<eigenIJVector> ind1 = smallIndexMap[ind];
//             Matrix oida ( (coneone_n+1 ), (coneone_n+1 ) );
//             oida.setZero();
//             for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
//                 oida ( ind1[k] ( 0 ),ind1[k] ( 1 ) )+=  1;
//                 if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
//                     oida ( ind1[k] ( 1 ),ind1[k] ( 0 ) )+=  1;
//                 }
//             }
// //             std::cout << " oida " << counter << std::endl << oida << std::endl;
//             M_Matrix::t Lwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( oida.data(), monty::shape ( ( int ) (coneone_n+1 ), ( int ) (coneone_n+1 ) ) ) ) );
//             bige = M_Expr::add ( bige,M_Expr::mul ( nu->index ( counter ),Lwrap ) );
//             ++counter;
//         }
// //         Eigen::SelfAdjointEigenSolver<Matrix> eig0(H);
// //         const double meanentry((eig0.eigenvectors() * eig0.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose()).mean());
// //         M->constraint ( M_Expr::sub ( M_Expr::mul(meanentry,bige),MM ),M_Domain::equalsTo ( 0.0 ) );
//         M->constraint ( M_Expr::sub ( bige,MM ),M_Domain::equalsTo ( 0.0 ) );
// 
//         //      the matrix that is such that the coefficients of the polynomial are mapped to the matrix
//         M_Variable::t MM2 = M->variable ( "SemiCone2", M_Domain::inPSDCone ( ( int ) (conetwo_n+1 ) ) ); // the moment matrix
// //         the coefficients
// 
// 
// 
// //      find the coefficients of the big coefficient vector in the matrix of the quadratic form
//         counter = 0;
//         M_Expression::t bige_2 = M_Expr::zeros ( nint ( { ( int ) (conetwo_n+1 ), ( int ) (conetwo_n+1 )} ) );
//         for ( const auto &ind : mySet3 ) {
// //          find it in the map
//             const std::vector<eigenIJVector> ind1 = smallIndexMap_2[ind];
//             Matrix oida ( (conetwo_n+1 ), (conetwo_n+1 ) );
//             oida.setZero();
//             for ( Eigen::Index k = 0; k < ind1.size(); ++k ) {
//                 oida ( ind1[k] ( 0 ),ind1[k] ( 1 ) )+=  1;
//                 if ( ind1[k] ( 0 ) != ind1[k] ( 1 ) ) {
//                     oida ( ind1[k] ( 1 ),ind1[k] ( 0 ) )+=  1;
//                 }
//             }
// //             std::cout << " oidasmall " << counter << std::endl << oida << std::endl;
//             M_Matrix::t Lwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( oida.data(), monty::shape ( ( int ) (conetwo_n+1 ), ( int ) (conetwo_n+1 ) ) ) ) );
//             bige_2 = M_Expr::add ( bige_2,M_Expr::mul ( nu->index ( counter ),Lwrap ) );
//             ++counter;
//         }
// //         M->constraint ( M_Expr::sub ( M_Expr::mul(meanentry,bige_2),MM2 ),M_Domain::equalsTo ( 0.0 ) );
//         M->constraint ( M_Expr::sub ( bige_2,MM2 ),M_Domain::equalsTo ( 0.0 ) );
//         
//         return(EXIT_SUCCESS);
//     }
//     
//     
//      int  fillConstrainedDualObjective ( M_Model M,
//                                         M_Variable::t nu,
//                                         M_Variable::t eta,
//                                         M_Variable::t epsilon,
//                                         const Matrix& H,
//                                         const Matrix& S,
//                                         const Vector& rhsS,
//                                         const Matrix& U,
//                                         const Vector& rhsU ) const {
//         
//         
// 
//         assert ( S.rows() == rhsS.size() );
//         assert ( U.rows() == rhsU.size() );
//         assert ( U.cols() == H.rows() );
//         assert ( S.cols() == H.rows() );
// 
//         const int Urow ( U.rows() );
//         const int Srow ( S.rows() );
// 
// 
//         M_Variable::t u = M->variable ( "obj", 1, M_Domain::greaterThan ( 0.0 ) );
//         M_Expression::t theta = M_Expr::vstack ( eta,nu,epsilon );
// 
// 
//         //         now the constraints
//         const Matrix equVec = S * H;
//         const Matrix inequVec = U * H;
//         
//          Eigen::SelfAdjointEigenSolver<Matrix> eig0(H);
//         Matrix Hinvsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
//         Matrix Hsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
// //         UU is the matrix (S sqrt(H) \\ sqrt(H^-1) \\ -u sqrt(H))
// //         need to form in transpose so we can give it to matrix::dense
// 
//         Matrix UU ( Srow + H.rows() + Urow, H.cols() );
//         UU << S * Hsqrt,
//                 Hinvsqrt, 
//                 -U * Hsqrt;
//                 
//                 std::cout << UU << std::endl;
// 
// 
//         Vector rhss = rhsS;
//         Vector rhsu = rhsU;
// 
//         const auto equVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhss.data(),  monty::shape ( rhss.size() ) ) );
//         const auto inequVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhsu.data(),  monty::shape ( rhsu.size() ) ) );
//         
// 
//         
//         M_Matrix::t UUwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( UU.data(), monty::shape ( UU.cols(), UU.rows() ) ) ) );
//         M->constraint ( M_Expr::vstack ( 0.5,u, M_Expr::mul ( UUwrap,theta  ) ), M_Domain::inRotatedQCone() );
//         M->objective ( ObjectiveSense::Maximize, M_Expr::add ( M_Expr::mul ( -0.5,u ),M_Expr::sub ( M_Expr::dot ( eta,equVecRhsWrap ),M_Expr::dot ( epsilon,inequVecRhsWrap ) ) ) );
//         return(EXIT_SUCCESS);
//     }
//     
//     
// //     int  fillConstrainedDualObjective ( M_Model M,
// //                                         M_Variable::t nu,
// //                                         M_Variable::t eta,
// //                                         M_Variable::t epsilon,
// //                                         const Matrix& H,
// //                                         const Matrix& S,
// //                                         const Vector& rhsS,
// //                                         const Matrix& U,
// //                                         const Vector& rhsU ) const {
// //         
// //         
// // 
// //         assert ( S.rows() == rhsS.size() );
// //         assert ( U.rows() == rhsU.size() );
// //         assert ( U.cols() == H.rows() );
// //         assert ( S.cols() == H.rows() );
// // 
// //         const int Urow ( U.rows() );
// //         const int Srow ( S.rows() );
// // 
// // 
// //         M_Variable::t u = M->variable ( "obj", 1, M_Domain::greaterThan ( 0.0 ) );
// // //         M_Expression::t theta = M_Expr::vstack ( eta,nu,epsilon );
// // 
// // 
// //         //         now the constraints
// // //         const Matrix equVec = S * H;
// // //         const Matrix inequVec = U * H;
// //         
// //          Eigen::SelfAdjointEigenSolver<Matrix> eig0(H);
// //         Matrix Hinvsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
// //         Matrix Hinv = eig0.eigenvectors() * eig0.eigenvalues().cwiseInverse().asDiagonal() * eig0.eigenvectors().transpose();
// //         Matrix Hsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
// // //         UU is the matrix (S sqrt(H) \\ sqrt(H^-1) \\ -u sqrt(H))
// // //         need to form in transpose so we can give it to matrix::dense
// // 
// // //         Matrix UU ( Srow + H.rows() + Urow, H.cols() );
// // //         UU << S * Hsqrt,
// // //                 Hinvsqrt, 
// // //                 -U * Hsqrt;
// //                 
// // //                 std::cout << UU << std::endl;
// //         
// //         std::cout << " S " << std::endl << S << std::endl;
// //         std::cout << " U " << std::endl << U << std::endl;
// //         std::cout << " H " << std::endl << H << std::endl;
// //         std::cout << " S " << std::endl << S << std::endl;
// //         std::cout << " rhsS " << std::endl << rhsS << std::endl;
// //         std::cout << " rhsU " << std::endl << rhsU << std::endl;
// //         std::cout << " Hinvsqrt " << std::endl << Hinvsqrt << std::endl;
// //         std::cout << " Hsqrt " << std::endl << Hsqrt << std::endl;
// //         std::cout << " Hinv " << std::endl << Hinv << std::endl;
// // 
// //         M_Variable::t xx = M->variable ( "coeff", initN, M_Domain::unbounded() );
// //         Vector rhss = rhsS;
// //         Vector rhsu = rhsU;
// // 
// //         const auto equVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhss.data(),  monty::shape ( rhss.size() ) ) );
// //         const auto inequVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhsu.data(),  monty::shape ( rhsu.size() ) ) );
// //         
// //         Matrix Sunconst = S;
// //         M_Matrix::t SWrap_t = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( Sunconst.data(), monty::shape ( Sunconst.cols(), Sunconst.rows() ) ) ) );
// //         Matrix Uunconst = U;
// //         M_Matrix::t UWrap_t = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( Uunconst.data(), monty::shape ( Uunconst.cols(), Uunconst.rows() ) ) ) );
// //         
// //         M_Matrix::t Hinvsqrtwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( Hinvsqrt.data(), monty::shape ( Hinvsqrt.cols(), Hinvsqrt.rows() ) ) ) );
// //         M_Matrix::t Hsqrtwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( Hsqrt.data(), monty::shape ( Hsqrt.cols(), Hsqrt.rows() ) ) ) );
// //         M_Matrix::t Hinvwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( Hinv.data(), monty::shape ( Hinv.cols(), Hinv.rows() ) ) ) );
// //         
// //          M->constraint ( M_Expr::sub(M_Expr::add(M_Expr::mul(SWrap_t,eta),M_Expr::sub(M_Expr::mul(Hinvwrap,nu),M_Expr::mul(UWrap_t,epsilon))),xx), M_Domain::equalsTo(0.0) );
// //         
// // 
// //         
// // //         M_Matrix::t UUwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( UU.data(), monty::shape ( UU.cols(), UU.rows() ) ) ) );
// //         M->constraint ( M_Expr::vstack ( 0.5,u, M_Expr::mul ( Hsqrtwrap,xx  ) ), M_Domain::inRotatedQCone() );
// //         M->objective ( ObjectiveSense::Maximize, M_Expr::add ( M_Expr::mul ( -0.5,u ),M_Expr::sub ( M_Expr::dot ( eta,equVecRhsWrap ),M_Expr::dot ( epsilon,inequVecRhsWrap ) ) ) );
// //         return(EXIT_SUCCESS);
// //     }
//     
// 
//     int  fillEqualityConstrainedDualObjective ( M_Model M,
//             M_Variable::t nu,
//             M_Variable::t eta,
//             const Matrix& H,
//             const Matrix& S,
//             const Vector& rhsS ) const {
//         
//         
// 
//         assert ( S.rows() == rhsS.size() );
//         assert ( S.cols() == H.rows() );
//         assert ( S.cols() == nu->getSize() );
//         assert ( S.rows() == eta->getSize() );
// 
//         const int Srow ( S.rows() );
// 
// 
//         M_Variable::t u = M->variable ( "obj", 1, M_Domain::greaterThan ( 0.0 ) );
//         M_Expression::t theta = M_Expr::vstack ( eta,nu );
// 
// 
//         //         now the constraints
//         const Matrix equVec = S * H;
//         Eigen::SelfAdjointEigenSolver<Matrix> eig0(H);
//         Matrix Hinvsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
//         Matrix Hsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
// //         UU is the matrix (S sqrt(H) \\ sqrt(H^-1) \\ -u sqrt(H))
// //         need to form in transpose so we can give it to matrix::dense
//         
//         std::cout << "eigenvectors " << eig0.eigenvalues().transpose() << std::endl;
// 
//         Matrix UU ( Srow + H.rows(), H.cols() );
//         UU << S*Hsqrt,
//               Hinvsqrt;
// 
//         Vector rhss = rhsS;
// 
// // 
//         const auto equVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhss.data(),  monty::shape ( rhss.size() ) ) );
//         M_Matrix::t UUwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( UU.data(), monty::shape ( UU.cols(), UU.rows() ) ) ) );
//         M->constraint ( M_Expr::vstack ( 0.5,u, M_Expr::mul ( UUwrap,theta  ) ), M_Domain::inRotatedQCone() );
//         M->objective ( ObjectiveSense::Maximize, M_Expr::add ( M_Expr::mul ( -0.5,u ),M_Expr::dot ( eta,equVecRhsWrap ) ) );
//         return(EXIT_SUCCESS);
//     }
// 
//     int  fillInequalityConstrainedDualObjective ( M_Model M,
//             M_Variable::t nu,
//             M_Variable::t epsilon,
//             const Matrix& H,
//             const Matrix& U,
//             const Vector& rhsU ) const {
//         
//         
// 
//         assert ( U.rows() == rhsU.size() );
//         assert ( U.cols() == H.rows() );
// 
//         const int Urow ( U.rows() );
// 
// 
//         M_Variable::t u = M->variable ( "obj", 1, M_Domain::greaterThan ( 0.0 ) );
//         M_Expression::t theta = M_Expr::vstack ( nu,epsilon );
//         
//         Eigen::SelfAdjointEigenSolver<Matrix> eig0(H);
//         Matrix Hinvsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
//         Matrix Hsqrt = eig0.eigenvectors() * eig0.eigenvalues().cwiseSqrt().asDiagonal() * eig0.eigenvectors().transpose();
// //         UU is the matrix (S sqrt(H) \\ sqrt(H^-1) \\ -u sqrt(H))
// //         need to form in transpose so we can give it to matrix::dense
// 
//         Matrix UU ( H.rows() + Urow, H.cols() );
//         UU << Hinvsqrt, 
//                 -U * Hsqrt;
// 
// 
//         Vector rhsu = rhsU;
// 
//         const auto inequVecRhsWrap = std::shared_ptr<  M_ndarray_1 > ( new  M_ndarray_1 ( rhsu.data(),  monty::shape ( rhsu.size() ) ) );
//         M_Matrix::t UUwrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2 > ( new M_ndarray_2 ( UU.data(), monty::shape ( UU.cols(), UU.rows() ) ) ) );
//         M->constraint ( M_Expr::vstack ( 0.5,u, M_Expr::mul ( UUwrap,theta  ) ), M_Domain::inRotatedQCone() );
//         M->objective ( ObjectiveSense::Maximize, M_Expr::sub ( M_Expr::mul ( -0.5,u ),M_Expr::dot ( epsilon,inequVecRhsWrap ) ) );
//         return(EXIT_SUCCESS);
//     }


} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
