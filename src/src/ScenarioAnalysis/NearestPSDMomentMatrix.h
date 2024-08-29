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
#ifndef RRCA_SCENARIOANALYSIS_NEARESTPSDMOMENTMATRIX_H_
#define RRCA_SCENARIOANALYSIS_NEARESTPSDMOMENTMATRIX_H_



namespace RRCA
{
namespace SCENARIOANALYSIS
{
   
       /*
   *    \brief attempts to find a psd matrix with the same moment structure as the input matrix, assuming canonical lexicographic order
   *  n denotes the order of the basis (so that the moment matrix has hhighest order 2n) and d the dimension
   */
class NearestPSDMomentMatrix
{
    
public:
    NearestPSDMomentMatrix( unsigned int d_, unsigned int n_, const Matrix& A_ ) : 
        d ( d_ ),n ( n_ ),
        initN ( binomialCoefficient( n_+d_, d_ ) ),
        myIndex ( d_,n_ ),
        A ( A_ ) {
        assert ( A.rows() == initN );
        assert ( A.cols() == initN );
//      compute the indices of the flat extension, it will contain these of the input matrix


        const auto &mySet = myIndex.get_MultiIndexSet();
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
                auto it = indexMap.find ( newIndex );
                if ( it != indexMap.end() )
                    it->second.push_back ( ij );
                else
                    indexMap[newIndex].push_back ( ij );
            }
        }
    }
    
    Matrix getClosestPSDMatrix() const {
        return(MMM);
    }

    int computePSDApproximation(double scal = 1.0) {

        M_Model M = new mosek::fusion::Model ( "NearestPSDMomentMatrix" );
//         M->setLogHandler ( [=] ( const std::string & msg ) {
//             std::cout << msg << std::flush;
//         } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );

        M_Variable::t MM = M->variable ( M_Domain::inPSDCone ( ( int ) initN ) ); // the moment matrix
        M_Variable::t uu = M->variable( "uu", M_Domain::greaterThan(0.0));
        Matrix oida___ = A/scal;

        const M_Matrix::t Awrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( oida___.data(), monty::shape ( (int)initN, (int)initN ) ) ));

//      ensure that we are dealing with a moment matrix, by setting all the entries that refer to the same moments equal to each other
        for ( const auto &ind1 : indexMap ) {
            if ( ind1.second.size() >1 ) {
                for ( Eigen::Index k = 1; k < ind1.second.size(); ++k ) {
                    M->constraint ( M_Expr::sub ( MM->index ( ind1.second[0] ( 0 ),ind1.second[0] ( 1 ) ),MM->index ( ind1.second[k] ( 0 ),ind1.second[k] ( 1 ) ) ),M_Domain::equalsTo ( 0.0 ) );
                }
            }
        }
        M->constraint (MM->index ( 0,0), M_Domain::equalsTo(1.0/scal));
//         M_Variable::t beta = M->variable(initN*initN,M_Domain::unbounded()); 
//         M_Variable::t betaplus = M->variable(initN*initN,M_Domain::greaterThan(0.0)); 
//         M_Variable::t betaminus = M->variable(initN*initN,M_Domain::greaterThan(0.0)); 
//         M->constraint (M_Expr::sub(beta,M_Expr::flatten(M_Expr::sub(MM,Awrap))) , M_Domain::equalsTo(0.0));
//         M->constraint (M_Expr::sub(beta,M_Expr::sub(betaplus,betaminus)) , M_Domain::equalsTo(0.0));
        
        M->constraint(M_Expr::vstack(uu, M_Expr::flatten(M_Expr::sub(MM,Awrap))), M_Domain::inQCone()); 

        M->objective ( mosek::fusion::ObjectiveSense::Minimize, uu);
//         M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::sum(M_Expr::add(betaplus,betaminus)));
        
        M->solve();
        if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
            const auto sol = std::shared_ptr<M_ndarray_1> ( new M_ndarray_1( * ( MM->level() ) ) );
            const Eigen::Map<Matrix> auxmat ( sol->raw(), A.cols() ,A.cols() );
            MMM = auxmat*scal;
//             H = auxmat;
//             h = H.reshaped();
//             std::cout << h.transpose() << std::endl;
        } else {
            std::cout << "infeasible  " <<  std::endl; 
            return ( EXIT_FAILURE );
        }

        return ( EXIT_SUCCESS );
    }
    private:
    const  unsigned int d;
    const  unsigned int n;
    const  unsigned int initN;
    const RRCA::MultiIndexSet<iVector> myIndex; // index set that contains the coefficients of the canonical basis that generates the flat extension of H (through outer product)
    std::map<iVector, std::vector<ijVector>, RRCA::FMCA_Compare<iVector> > indexMap;

    const Matrix& A;
    Matrix MMM;


};
    



} // namespace SCENARIOANALYSIS
}  // namespace RRCA
#endif
