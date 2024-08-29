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
#ifndef RRCA_ROWREDUCE_H_
#define RRCA_ROWREDUCE_H_


namespace RRCA {






// Matrix traits: This describes how a matrix is accessed. By
// externalizing this information into a traits class, the same code
// can be used both with native arrays and matrix classes. To use the
// default implementation of the traits class, a matrix type has to
// provide the following definitions as members:
//
// * typedef ... index_type;
//   - The type used for indexing (e.g. size_t)
// * typedef ... value_type;
//   - The element type of the matrix (e.g. double)
// * index_type min_row() const;
//   - returns the minimal allowed row index
// * index_type max_row() const;
//   - returns the maximal allowed row index
// * index_type min_column() const;
//   - returns the minimal allowed column index
// * index_type max_column() const;
//   - returns the maximal allowed column index
// * value_type& operator()(index_type i, index_type k)
//   - returns a reference to the element i,k, where
//     min_row() <= i <= max_row()
//     min_column() <= k <= max_column()
// * value_type operator()(index_type i, index_type k) const
//   - returns the value of element i,k
//
// Note that the functions are all inline and simple, so the compiler
// should completely optimize them away.
template<typename Derived>
struct matrix_traits
{
  typedef typename Eigen::MatrixBase<Derived> MatrixType;
  typedef Index index_type;
  typedef Scalar value_type;
  static index_type min_row(MatrixType const& A)
  { return 0; }
  static index_type max_row(MatrixType const& A)
  { return A.rows()-1; }
  static index_type min_column(MatrixType const& A)
  { return 0; }
  static index_type max_column(MatrixType const& A)
  { return A.cols()-1; }
  static value_type& element(MatrixType& A, index_type i, index_type k)
  { return A(i,k); }
  static value_type element(MatrixType const& A, index_type i, index_type k)
  { return A(i,k); }
};


// Swap rows i and k of a matrix A
// Note that due to the reference, both dimensions are preserved for
// built-in arrays
template<typename Derived>
 void swap_rows(Eigen::MatrixBase<Derived>& A,
                 typename matrix_traits<Derived>::index_type i,
                 typename matrix_traits<Derived>::index_type k)
{
  matrix_traits<Derived> mt;
  typedef typename matrix_traits<Derived>::index_type index_type;

  // check indices
  assert(mt.min_row(A) <= i);
  assert(i <= mt.max_row(A));

  assert(mt.min_row(A) <= k);
  assert(k <= mt.max_row(A));

  for (index_type col = mt.min_column(A); col <= mt.max_column(A); ++col)
    std::swap(mt.element(A, i, col), mt.element(A, k, col));
}

// divide row i of matrix A by v
template<typename Derived>
 void divide_row(Eigen::MatrixBase<Derived>& A,
                  typename matrix_traits<Derived>::index_type i,
                  typename matrix_traits<Derived>::value_type v)
{
  matrix_traits<Derived> mt;
  typedef typename matrix_traits<Derived>::index_type index_type;

  assert(mt.min_row(A) <= i);
  assert(i <= mt.max_row(A));

  assert(v != 0);

  for (index_type col = mt.min_column(A); col <= mt.max_column(A); ++col)
    mt.element(A, i, col) /= v;
}

// in matrix A, add v times row k to row i
template<typename Derived>
 void add_multiple_row(Eigen::MatrixBase<Derived>& A,
                  typename matrix_traits<Derived>::index_type i,
                  typename matrix_traits<Derived>::index_type k,
                  typename matrix_traits<Derived>::value_type v)
{
  matrix_traits<Derived> mt;
  typedef typename matrix_traits<Derived>::index_type index_type;

  assert(mt.min_row(A) <= i);
  assert(i <= mt.max_row(A));

  assert(mt.min_row(A) <= k);
  assert(k <= mt.max_row(A));

  for (index_type col = mt.min_column(A); col <= mt.max_column(A); ++col)
    mt.element(A, i, col) += v * mt.element(A, k, col);
}

// convert A to reduced row echelon form
template<typename Derived>
 void to_reduced_row_echelon_form(Eigen::MatrixBase<Derived>& A)
{
  matrix_traits<Derived> mt;
  typedef typename matrix_traits<Derived>::index_type index_type;

  index_type lead = mt.min_row(A);

  for (index_type row = mt.min_row(A); row <= mt.max_row(A); ++row)
  {
    if (lead > mt.max_column(A))
      return;
    index_type i = row;
    while (mt.element(A, i, lead) == 0)
    {
      ++i;
      if (i > mt.max_row(A))
      {
        i = row;
        ++lead;
        if (lead > mt.max_column(A))
          return;
      }
    }
    swap_rows(A, i, row);
    divide_row(A, row, mt.element(A, row, lead));
    for (i = mt.min_row(A); i <= mt.max_row(A); ++i)
    {
      if (i != row)
        add_multiple_row(A, i, row, -mt.element(A, i, lead));
    }
  }
}

    /*
   *   \brief Returns the input matrix C in row echelon form. Like the matlab rref function
   */
template <typename Derived>
int rowReduce(Eigen::MatrixBase<Derived> &C){
    to_reduced_row_echelon_form<Derived>(C);
    return(EXIT_SUCCESS);
}


} // namespace RRCA
#endif
