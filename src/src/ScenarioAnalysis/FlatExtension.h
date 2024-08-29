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
#ifndef RRCA_SCENARIOANALYSIS_FLATEXTENSION_H_
#define RRCA_SCENARIOANALYSIS_FLATEXTENSION_H_



namespace RRCA
{
namespace SCENARIOANALYSIS
{
    
        
   
       /*
   *    \brief attempts to solve for a flat extension starting from a problem of dimension d and order initn (so that A has highest order smaller or equal tha initn squared)
   *  addn describes how many additional orders are added to initn. A is allowed to be moment matrix of order smaller or equal than initn
   */
class FlatExtension
{




public:
    FlatExtension( unsigned int d_, unsigned int initn_, unsigned int addn_, const Matrix& A_ ) : d ( d_ ),n ( initn_ ),
        initN ( binomialCoefficient( initn_+d_, d_ ) ),
        extenN ( binomialCoefficient( initn_+ addn_ + d_, d_ ) ),
        myIndex ( d_,initn_ ),
        myFlatIndex ( d_,initn_+addn_ ),
        A ( A_ ),
        flatA(extenN,extenN) ,
        flatAEigenVals(extenN){
//         assert ( A.rows() == initN );
//         assert ( A.cols() == initN );
//      compute the indices of the flat extension, it will contain these of the input matrix


        const auto &mySet = myFlatIndex.get_MultiIndexSet();
//      copy into vector for simplicity
        std::vector<iVector> inter;
        for ( const auto &ind1 : mySet ) {
            inter.push_back ( ind1 );
        }
        for ( int j = 0; j < inter.size(); ++j ) {
            for ( int i = 0; i <= j; ++i ) {
                iVector newIndex = inter[i] + inter[j];
                ijVector ij;
                ij << i, j;
                auto it = flatIndexMap.find ( newIndex );
                if ( it != flatIndexMap.end() )
                    it->second.push_back ( ij );
                else
                    flatIndexMap[newIndex].push_back ( ij );
            }
        }
    }
    
    bool isFlat() const {
//         Eigen::SelfAdjointEigenSolver<Matrix> eigensolver ( flatA, Eigen::EigenvaluesOnly );
//         std::cout << "eigenvalues " << std::endl;
//         std::cout << eigensolver.eigenvalues().transpose() << std::endl;
//      perform pivoted cholesky
        PivotedCholeskyDecompositon<Matrix> piv;
        piv.compute(flatA, 1.e-8);

        if(piv.matrixL().cols() > initN){
            return(false);
        }
        return(true);
    }
    Matrix getFlatA() const {
        return(flatA);
    }

    int computeHankelExtension(double eps = 0.0, double scal = 1.0) {
        assert(eps>=0.0);
        M_Model M = new mosek::fusion::Model ( "HankelExtension" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } ); 

        M_Variable::t MM = M->variable ( M_Domain::inPSDCone ( ( int ) extenN ) ); // the moment matrix
        M_Variable::t D = MM->slice ( monty::new_array_ptr<int,1> ( { ( int ) initN, ( int ) initN} ), monty::new_array_ptr<int,1> ( { ( int ) extenN, ( int ) extenN} ) );
        
        
//         double const scal = A.rightCols(A.cols()-1).cwiseAbs().mean(); 
        // double const scal = 1.0; // looks at the average moment (excpet for moment 0)
//         std::cout << "scal " << scal << std::endl;
        Matrix oida___ = A/scal;

//  fill moment matrix
//  M ={{A,B},{B',D}}
//  the conditions are A w= B and w'Aw=D
//         std::cout << " now writing out " << std::endl;
//      now distinguish between strict equality or nonstrict
        if(eps>0.0){ //nonstrict
//             in any case keep the one
            M->constraint ( M_Expr::sub ( MM->index ( 0,0 ),oida___ ( 0,0 ) ),M_Domain::equalsTo ( 0.0 ) );
            M_Variable::t AA = MM->slice ( monty::new_array_ptr<int,1> ( { 0, 0} ), monty::new_array_ptr<int,1> ( { ( int ) A.rows(), ( int ) A.cols()}) );
            const M_Matrix::t oida___twrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2 ( oida___.data(), monty::shape ( oida___.cols(), oida___.rows() ) ) ) );
            M_Variable::t uu = M->variable ("uu", 1, M_Domain::lessThan(eps)  );
            M->constraint(M_Expr::vstack(uu, M_Expr::flatten(M_Expr::sub (AA,oida___twrap ))), M_Domain::inQCone());
            
        } else {
            for ( unsigned int i = 0; i < A.cols() ; ++i ) {
                for ( unsigned int j = 0; j <= i; ++j ) {
//              A ought to be equal to the input matrix
                    M->constraint ( M_Expr::sub ( MM->index ( i,j ),oida___ ( j,i ) ),M_Domain::equalsTo ( 0.0 ) );
                }
            }
        }


//      ensure that we are dealing with a moment matrix, by setting all the entries that refer to the same moments equal to each other
        for ( const auto &ind1 : flatIndexMap ) {
            if ( ind1.second.size() >1 ) {
                for ( Eigen::Index k = 1; k < ind1.second.size(); ++k ) {
//                     std::cout << " index " << ind1.second[k].transpose() << std::endl;
                    M->constraint ( M_Expr::sub ( MM->index ( ind1.second[0] ( 0 ),ind1.second[0] ( 1 ) ),MM->index ( ind1.second[k] ( 0 ),ind1.second[k] ( 1 ) ) ),M_Domain::equalsTo ( 0.0 ) );
                }
            }

//             std::cout << "\n--------" << std::endl;
        }


//         M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::mul ( 100000000.0,M_Expr::sum ( D->diag() ) ) );
        M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::sum ( D->diag()  ));
        
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal && M->getDualSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
            auto resMat = * ( MM->level() );
//             const Eigen::Map<Matrix> oida ( resMat.raw(), extenN ,extenN );
            flatA = Eigen::Map<Matrix> ( resMat.raw(), extenN ,extenN )*scal;
        } else {
            std::cout << "flat extension infeasible " << std::endl;
            return(EXIT_FAILURE);
        }

        return ( EXIT_SUCCESS );
    }
    private:
    const  unsigned int d;
    const  unsigned int n;
    const  unsigned int initN;
    const  unsigned int extenN;
    const RRCA::MultiIndexSet<iVector> myIndex; // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
    const RRCA::MultiIndexSet<iVector> myFlatIndex;
    std::map<iVector, std::vector<ijVector>, RRCA::FMCA_Compare<iVector> > indexMap;
    std::map<iVector, std::vector<ijVector>, RRCA::FMCA_Compare<iVector> > flatIndexMap;


    iMatrix canInputBasis;
    iMatrix flatMoments; // collects all the different powers that are in the flat extension of H. There are N (N+1)/2 in the lower triangle, where N=extenN
    iMatrix flatIndices;

    std::vector<iVector> flatMomentVec;

    const Matrix& A;
    Matrix flatA;
    Vector flatAEigenVals;


};
    



} // namespace SCENARIOANALYSIS
}  // namespace RRCA
#endif
