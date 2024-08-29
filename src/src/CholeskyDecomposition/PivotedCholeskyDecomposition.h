// ////////////////////////////////////////////////////////////////////////////////
// // This file is part of RRCA, the Roadrunner Covariance Analsys package.      //
// //                                                                            //
// // Copyright (c) 2021, Michael Multerer and Paul Schneider                    //
// //                                                                            //
// // All rights reserved.                                                       //
// //                                                                            //
// // This source code is subject to the BSD 3-clause license and without        //
// // any warranty, see <https://github.com/muchip/RRCA> for further             //
// // information.                                                               //
// ////////////////////////////////////////////////////////////////////////////////
// #ifndef RRCA_CHOLESKYDECOMPOSITION_PIVOTEDCHOLESKYDECOMPOSITION_H_
// #define RRCA_CHOLESKYDECOMPOSITION_PIVOTEDCHOLESKYDECOMPOSITION_H_

// namespace RRCA {

// template <typename KernelMatrix>
// class PivotedCholeskyDecompositon
//     : public CholeskyDecompositionBase<
//       PivotedCholeskyDecompositon<KernelMatrix>, KernelMatrix> {
// public:
//     typedef CholeskyDecompositionBase<PivotedCholeskyDecompositon, KernelMatrix>
//     Base;
//     // get types from base class
//     using value_type = typename Base::value_type;
//     using kernelMatrix = typename Base::kernelMatrix;
//     // get member variables from base class
//     using Base::Bmatrix_;
//     using Base::indices_;
//     using Base::info_;
//     using Base::Lmatrix_;
//     using Base::tol_;
//     PivotedCholeskyDecompositon() {}
//     // non-void constructor
//     PivotedCholeskyDecompositon ( const kernelMatrix &C, value_type tol ) {
//         compute ( C, tol );
//     }
//     /*
//      *   \brief computes the pivoted Cholesky decomposition with diagonal
//      *          pivoting
//      */
//     void compute ( const kernelMatrix &C, value_type tol, Eigen::Index piv_limit = 0, Eigen::Index step_limit = 0 ) {
//         Vector D;
//         Eigen::Index pivot = 0;
//         Eigen::Index actBSize = 0;
//         Eigen::Index cols = piv_limit ? piv_limit : C.cols();
//         Eigen::Index limit = step_limit ? step_limit : C.cols();
//         Eigen::Index rows = C.rows();
//         value_type tr = 0;
//         tol_ = tol;
//         // compute the diagonal and the trace
//         D = C.diagonal();
//         if ( D.minCoeff() < 0 ) {
//             info_ = 1;
//             return;
//         }
//         tr = D.sum();
//         // we guarantee the error tr(A-LL^T)/tr(A) < tol
//         tol *= tr;
//         // perform pivoted Cholesky decomposition
//         Eigen::Index step = 0;
//         while ( ( step < cols ) && ( tol < tr ) && ( step < limit ) ) {
//             // check memory requirements
//             if ( actBSize - 1 <= step ) {
//                 actBSize += allocBSize;
//                 Lmatrix_.conservativeResize ( rows, actBSize );
//                 indices_.conservativeResize ( actBSize );
//             }
//             D.head ( cols ).maxCoeff ( &pivot );
//             indices_ ( step ) = pivot;
//             // get new column from C
//             Lmatrix_.col ( step ) = C.col ( pivot );
//             // update column with the current matrix Lmatrix_
//             Lmatrix_.col ( step ) -= Lmatrix_.block ( 0, 0, rows, step ) *
//                                      Lmatrix_.row ( pivot ).head ( step ).transpose();
//             if ( Lmatrix_ ( pivot, step ) <= 0 ) {
//                 info_ = 2;
//                 std::cout << "breaking here\n";
//                 break;
//             }
//             Lmatrix_.col ( step ) /= sqrt ( Lmatrix_ ( pivot, step ) );
//             // update the diagonal and the trace
//             D.array() -= Lmatrix_.col ( step ).array().square();
//             // compute the trace of the Schur complement
//             tr = D.sum();
//             ++step;
//         }

//         if ( tr < 0 )
//             info_ = 2;
//         else
//             info_ = 0;
//         // crop L, indices to their actual size
//         Lmatrix_.conservativeResize ( rows, step );
//         indices_.conservativeResize ( step );
//         return;
//     }

// //   computes L matrix for given pivots
//     void compute ( const kernelMatrix &C, const Eigen::Matrix<Index, Eigen::Dynamic, 1u>& pivots ) {
//         // Vector D;
//         Eigen::Index pivot = 0;
//         const Eigen::Index rows = C.rows();

//         Lmatrix_.resize ( rows, pivots.size() );
//         indices_.resize ( pivots.size() );
//         // perform pivoted Cholesky decomposition
//         Eigen::Index step = 0;
//         for ( unsigned int i = 0; i < pivots.size(); ++i ) {
//             indices_ ( i ) = pivots ( i );
//             pivot = pivots ( i ); 
//             // get new column from C
//             Lmatrix_.col ( step ) = C.col ( pivot );
//             // update column with the current matrix Lmatrix_
//             Lmatrix_.col ( step ) -= Lmatrix_.block ( 0, 0, rows, step ) *
//                                      Lmatrix_.row ( pivot ).head ( step ).transpose();
//             if ( Lmatrix_ ( pivot, step ) <= 0 ) {
//                 info_ = 2;
//                 std::cout << "breaking here\n";
//                 break;
//             }
//             Lmatrix_.col ( step ) /= sqrt ( Lmatrix_ ( pivot, step ) );
//             // update the diagonal and the trace
//             // D.array() -= Lmatrix_.col ( step ).array().square();
//             ++step;
//         }


//             info_ = 0;
//         // crop L, indices to their actual size

//         return;
//     }
// };

// }  // namespace RRCA
// #endif


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
#ifndef RRCA_CHOLESKYDECOMPOSITION_PIVOTEDCHOLESKYDECOMPOSITION_H_
#define RRCA_CHOLESKYDECOMPOSITION_PIVOTEDCHOLESKYDECOMPOSITION_H_

#include <iostream>
namespace RRCA {

template <typename KernelMatrix>
class PivotedCholeskyDecomposition
    : public CholeskyDecompositionBase<
      PivotedCholeskyDecomposition<KernelMatrix>, KernelMatrix> {
public:
    typedef CholeskyDecompositionBase<PivotedCholeskyDecomposition, KernelMatrix>
    Base;
    // get types from base class
    using value_type = typename Base::value_type;
    using kernelMatrix = typename Base::kernelMatrix;
    // get member variables from base class
    using Base::Bmatrix_;
    using Base::indices_;
    using Base::info_;
    using Base::Lmatrix_;
    using Base::tol_;
    PivotedCholeskyDecomposition() {}
    // non-void constructor
    PivotedCholeskyDecomposition ( const kernelMatrix &C, value_type tol ) {
        compute ( C, tol );
    }
    /*
     *   \brief computes the pivoted Cholesky decomposition with diagonal
     *          pivoting
     */
    void compute ( const kernelMatrix &C, value_type tol, Eigen::Index piv_limit = 0, Eigen::Index step_limit = 0 ) {
        Vector D;
        Eigen::Index pivot = 0;
        Eigen::Index actBSize = 0;
        Eigen::Index cols = piv_limit ? piv_limit : C.cols();
        Eigen::Index limit = step_limit ? step_limit : C.cols();
        Eigen::Index rows = C.rows();
        value_type tr = 0;
        tol_ = tol;
        std::cout << 31 << std::endl;
        std::cout << "C: " << std::endl; 
        std::cout << "C " <<C.full() << std::endl;  // interagire con c manda tutto all'aria
        std::cout << "C: " << std::endl; 
        // compute the diagonal and the trace
        D = C.diagonal();
        std::cout << 31.5 << std::endl;
        if ( D.minCoeff() < 0 ) {
            info_ = 1;
            return;
        }
        std::cout << 32 << std::endl;
        tr = D.sum();
        // we guarantee the error tr(A-LL^T)/tr(A) < tol
        tol *= tr;
        // perform pivoted Cholesky decomposition
        Eigen::Index step = 0;

        std::cout << 33 << std::endl;
        while ( ( step < cols ) && ( tol < tr ) && ( step < limit ) ) {
            // check memory requirements
            if ( actBSize - 1 <= step ) {
                actBSize += allocBSize;
                Lmatrix_.conservativeResize ( rows, actBSize );
                indices_.conservativeResize ( actBSize );
            }
            D.head ( cols ).maxCoeff ( &pivot );
            indices_ ( step ) = pivot;
            // get new column from C
            Lmatrix_.col ( step ) = C.col ( pivot );
            // update column with the current matrix Lmatrix_
            Lmatrix_.col ( step ) -= Lmatrix_.block ( 0, 0, rows, step ) *
                                     Lmatrix_.row ( pivot ).head ( step ).transpose();
            if ( Lmatrix_ ( pivot, step ) <= 0 ) {
                info_ = 2;
                std::cout << "breaking here\n";
                break;
            }
            Lmatrix_.col ( step ) /= sqrt ( Lmatrix_ ( pivot, step ) );
            // update the diagonal and the trace
            D.array() -= Lmatrix_.col ( step ).array().square();
            // compute the trace of the Schur complement
            tr = D.sum();
            ++step;
        }
        std::cout << 33 << std::endl;

        if ( tr < 0 )
            info_ = 2;
        else
            info_ = 0;
        std::cout << 34 << std::endl;
        // crop L, indices to their actual size
        Lmatrix_.conservativeResize ( rows, step );
        indices_.conservativeResize ( step );
        std::cout << 35 << std::endl;
        return;
    }

//   computes L matrix for given pivots
    void compute ( const kernelMatrix &C, const Eigen::Matrix<Index, Eigen::Dynamic, 1u>& pivots ) {
        // Vector D;
        Eigen::Index pivot = 0;
        const Eigen::Index rows = C.rows();

        Lmatrix_.resize ( rows, pivots.size() );
        indices_.resize ( pivots.size() );
        // perform pivoted Cholesky decomposition
        Eigen::Index step = 0;
        for ( unsigned int i = 0; i < pivots.size(); ++i ) {
            indices_ ( i ) = pivots ( i );
            pivot = pivots ( i ); 
            // get new column from C
            Lmatrix_.col ( step ) = C.col ( pivot );
            // update column with the current matrix Lmatrix_
            Lmatrix_.col ( step ) -= Lmatrix_.block ( 0, 0, rows, step ) *
                                     Lmatrix_.row ( pivot ).head ( step ).transpose();
            if ( Lmatrix_ ( pivot, step ) <= 0 ) {
                info_ = 2;
                std::cout << "breaking here\n";
                break;
            }
            Lmatrix_.col ( step ) /= sqrt ( Lmatrix_ ( pivot, step ) );
            // update the diagonal and the trace
            // D.array() -= Lmatrix_.col ( step ).array().square();
            ++step;
        }


            info_ = 0;
        // crop L, indices to their actual size

        return;
    }
};

}  // namespace RRCA
#endif
