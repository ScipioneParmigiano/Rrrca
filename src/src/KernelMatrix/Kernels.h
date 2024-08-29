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
#ifndef RRCA_KERNELMATRIX_KERNELS_H_
#define RRCA_KERNELMATRIX_KERNELS_H_

namespace RRCA {

  enum Kernels { gauss, linear, gammaexponential, rationalquadratic,cosine,inversemultiquadric,laplace,
    gaussop1,cosineop1,laplaceop1,inversemultiquadricop1,linearop1};

    inline static std::string kernelTypeString(const Kernels whichkernel){
      switch (whichkernel){
        case gauss : {
          return(std::string("gauss"));
        }
        case linear : {
          return(std::string("lin"));
        }
        case gammaexponential : {
          return(std::string("gamma_exp"));
        }
        case rationalquadratic : {
          return("ratquad");
        }
        case cosine : {
          return("cos");
        }
        case inversemultiquadric : {
          return("inv_mult_quad");
        }
        case laplace : {
          return("laplace");
        }
        case gaussop1 : {
          return("gaussop1");
        }
        case cosineop1 : {
          return("cosop1");
        }
        case linearop1 : {
          return("linop1");
        }
        case laplaceop1 : {
          return("laplaceop1");
        }
        case inversemultiquadricop1 : {
          return("inv_mult_quadop1");
        }
      }
    }

struct Matern32Kernel {
  double l = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  static double distance(double d){return(1. + sqrt(3) * d) * exp(-sqrt(3) * d);}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    double d = (x - y).norm() / l;
    return (1. + sqrt(3) * d) * exp(-sqrt(3) * d);
  }
};

struct Matern52Kernel {
  double l = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  static double distance(double d){return(1 + sqrt(5) * d + 5. / 3. * d * d) * std::exp(-sqrt(5) * d);}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    double d = (x - y).norm() / l;
    return (1 + sqrt(5) * d + 5. / 3. * d * d) * std::exp(-sqrt(5) * d);
  }
};

struct Matern72Kernel {
  double l = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  static double distance(double d){return(1. + sqrt(7) * d + 14. / 5 * d * d +
            7. * sqrt(7) / 15. * d * d * d) *
           exp(-sqrt(7) * d);}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    double d = (x - y).norm() / l;
    return (1. + sqrt(7) * d + 14. / 5 * d * d +
            7. * sqrt(7) / 15. * d * d * d) *
           exp(-sqrt(7) * d);
  }
};

struct Matern92Kernel {
  double l = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  inline static double distance(double d){return(1. + 3. * d + 27. / 7. * d * d + 18. / 7. * d * d * d +
            27. / 35. * pow(d, 4.) *
           exp(-3 * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    double d = (x - y).norm() / l;
    return (1. + 3. * d + 27. / 7. * d * d + 18. / 7. * d * d * d +
            27. / 35. * pow(d, 4.) *
           exp(-3 * d));
  }
};

struct MaternInfKernel {
  double l = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  static constexpr unsigned int PARMNUM = 1;
  static constexpr Kernels type = Kernels::gauss;
  inline static double distance(double d){return(std::exp(-0.5 * d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    double d = (x - y).norm() / l;
    return std::exp(-0.5 * d * d/static_cast<double>(x.size()));
  }
      inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm==0);
      return(1.0/4.0 + exp(input)); 
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm==0);  
    return(log(input-1.0/4.0)); 
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm==0);
      l = input;
  }
};

struct GaussianKernelOPlus1 {
  double l = 1;
  double c0 = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  static constexpr unsigned int PARMNUM = 2;
  static constexpr Kernels type = Kernels::gaussop1;
  inline static double distance(double d){return(std::exp(-0.5 * d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    double d = (x - y).norm() / l;
    return c0+std::exp(-0.5 * d * d/static_cast<double>(x.size()));
  }
      inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm==0 || whichParm==1);
        if(whichParm==0){
          return(1.0/4.0 + exp(input));
        }
      return(exp(input)); 
    }

  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm==0 || whichParm==1);
    if(whichParm==0){
      return(log(input-1.0/4.0));
    }  
    return(log(input)); 
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm==0 || whichParm==1);
      if(whichParm==0){
        l = input;
      } else {
        c0 = input;
      }
  }
};

struct LaplaceKernel {
  double l = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  static constexpr unsigned int PARMNUM = 1;
  static constexpr Kernels type = Kernels::laplace;
  // inline static double distance(double d){return(std::exp(-0.5 * d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    double d = (x - y).norm() / l;
    return std::exp(- d /sqrt(static_cast<double>(x.size())));
  }
      inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm==0);
      return(1.0/8.0 + exp(input)); 
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm==0);  
    return(log(input-1.0/8.0)); 
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm==0);
      l = input;
  }
};

struct LaplaceKernelOPlus1 {
  double l = 1;
  double c0 = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  static constexpr unsigned int PARMNUM = 2;
  static constexpr Kernels type = Kernels::laplaceop1;
  // inline static double distance(double d){return(std::exp(-0.5 * d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    double d = (x - y).norm() / l;
    return c0 + std::exp(- d /sqrt(static_cast<double>(x.size())));
  }
      inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm==0 || whichParm==1);
        if(whichParm==0){
          return(1.0/8.0 + exp(input));
        }
      return(exp(input)); 
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm==0||whichParm==1);
    if(whichParm==0){
      return(log(input-1.0/8.0));
    }  
    return(log(input)); 
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm==0||whichParm==1);
      if(whichParm==0){
        l = input;
      } else {
        c0 = input;
      }
  }
};

// l should be in the range (0,1]
struct GammaExponentialKernel {
  double l = 1;
  static constexpr double inf = 0.0;
  static constexpr double sup = 1.0;
  static constexpr unsigned int PARMNUM = 1;
  static constexpr Kernels type = Kernels::gammaexponential;
  inline static double distance(double d){return(std::exp(d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    // double d = (x - y).norm();
    return std::exp(-std::pow((x - y).squaredNorm()/static_cast<double>(x.size()),l));
  }
  
      inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm==0);
        return(exp(input)/(1.0+exp(input)));
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm==0);
      return(log(input/(1.0-input)));
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm==0);
      l = input;
  }
};





//   parsimonous versino of rational kernel with sigma = 1
struct RationalQuadraticKernel {
  double ell = 1;//parm1
  double alpha = 1;//parm2
  // static constexpr double inf = 0.0;
  // static constexpr double sup = 1.0;
  static constexpr unsigned int PARMNUM = 2;
  static constexpr Kernels type = Kernels::rationalquadratic;
  // inline static double distance(double d){return(std::exp(d * d));}
  template <typename Derived, typename otherDerived>
  inline double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return(std::pow(1.0+(x-y).squaredNorm()/(2.0*alpha*ell),-alpha));
  }

      inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm == 0 || whichParm == 1);
        if(whichParm == 0 || whichParm == 1){
          return(exp(input));
        }
//      must be parm 2
        return(INFINITY);
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm == 0 || whichParm == 1);
        if (whichParm == 0 || whichParm == 1){
          return(log(input));
        }
//      must be parm 2
        return(INFINITY);
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm == 0 || whichParm == 1);
      if (whichParm ==0){
          ell = input;
      }
      alpha = input;
  }
};

//   parsimonous versino of rational kernel with sigma = 1
struct CosineKernel {
  static constexpr unsigned int PARMNUM = 0;
  static constexpr Kernels type = Kernels::cosine;
  // inline static double distance(double d){return(std::exp(d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return(x.dot(y)/(x.norm()*y.norm()));
  }
// should never be called
      inline  double parmTransform(double input, unsigned int whichParm) const {
        return(INFINITY);
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    return(INFINITY);
  }
  void setParameter(double input, unsigned int whichParm){
      // return(0/0);
  }
};

struct CosineKernelOPlus1 {
  double c0 = 1;  
  static constexpr unsigned int PARMNUM = 1;
  static constexpr Kernels type = Kernels::cosineop1;
  // inline static double distance(double d){return(std::exp(d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return(c0 + x.dot(y)/(x.norm()*y.norm()));
  }
      inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm == 0);
        return(exp(input));
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm == 0);
        return(log(input));
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm == 0);
      c0 = input;
  }
};


struct LinearKernelOPlus1 {
  double c0 = 1;  
  static constexpr unsigned int PARMNUM = 1;
  static constexpr Kernels type = Kernels::linearop1;
  // inline static double distance(double d){return(std::exp(d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return(c0 + x.dot(y));
  }
      inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm == 0);
        return(exp(input));
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm == 0);
        return(log(input));
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm == 0);
      c0 = input;
  }
};

struct InverseMultiQuadricKernel {
  double l = 1;
  
  static constexpr unsigned int PARMNUM = 1;
  static constexpr Kernels type = Kernels::inversemultiquadric;
  // static constexpr double inf = 0.0;
  // static constexpr double sup = 1.0;
  // inline static double distance(double d){return(std::exp(-0.5 * d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return(1.0/std::sqrt((x - y).squaredNorm()+l));
  }
  inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm==0);
      return(exp(input)); 
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm==0);  
    return(log(input)); 
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm==0);
      l = input;
  }
};

struct InverseMultiQuadricKernelOPlus1 {
  double l = 1;
  double c0 = 1;
  static constexpr unsigned int PARMNUM = 2;
  static constexpr Kernels type = Kernels::inversemultiquadricop1;
  // static constexpr double inf = 0.0;
  // static constexpr double sup = 1.0;
  // inline static double distance(double d){return(std::exp(-0.5 * d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return(c0 + 1.0/std::sqrt((x - y).squaredNorm()+l));
  }
  inline  double parmTransform(double input, unsigned int whichParm) const {
        assert(whichParm==0||whichParm==1);

      return(exp(input)); 
    }
  inline  double parmInverseTransform(double input, unsigned int whichParm) const {
    assert(whichParm==0||whichParm==1);  
    return(log(input)); 
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm==0||whichParm==1);
      if(whichParm==0)
        l = input;
      else {
        c0 = input; 
      }
  }
};



// this kernel computes sample moments
// x and y here are time series
// it is a cross sectional kernel
struct CovarianceKernel {
  double l = 1;
  static constexpr double inf = -INFINITY;
  static constexpr double sup = INFINITY;
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return x.dot(y)/static_cast<double>(x.size());
  }
};

// this kernel computes sample moments
// x and y here are time series
struct RankOneKernel {
  std::function<double (double)> fun;
  static constexpr double inf = -INFINITY;
  static constexpr double sup = INFINITY;
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return fun(x.sum()) * fun(y.sum()); // need to reduce , while stayinv compatible with the interface
  }
};




using GaussianKernel = MaternInfKernel;


template<typename Kernel = GaussianKernel>
struct SumMaternKernel {
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> eigenVector;
    double l = 1;
//   static constexpr double inf = 0.0;
//   static constexpr double sup = 1.0;
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return((x-y).cwiseAbs().unaryExpr([&](typename Derived::value_type arg){
                              return(Kernel::distance(arg/l));
                            }).sum());
  }
};

struct SumGaussKernel {
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> eigenVector;
    double l = 1;
//   static constexpr double inf = 0.0;
//   static constexpr double sup = 1.0;
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return((((x-y).array().square()*(-0.5/(l*l))).exp()).mean());
  }
};

template<unsigned int j>
struct EuclideanPolynomialKernel {
    double l = 1;
//   static constexpr double inf = 0.0;
//   static constexpr double sup = 1.0;
//   inline static double distance(double d){return(std::exp(-0.5 * d * d));}
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
//     double d = (x - y).norm() / l;
    const double oida(x.dot(y)+l);
    return(std::pow(oida,j));
    }
};

struct LinearKernel {
    double l = 1;
    static constexpr unsigned int PARMNUM = 1;
    static constexpr Kernels type = Kernels::linear;
    
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
    return(l+x.dot(y));
    }
    
    inline  double parmTransform(double input,unsigned int whichParm) const {
      assert(whichParm==0);
      return(exp(input)); 
    }
  inline double parmInverseTransform(double input,unsigned int whichParm) const {
    assert(whichParm==0);
      return(log(input)); 
  }
  void setParameter(double input, unsigned int whichParm){
      assert(whichParm==0);
      l = input;
  }
};

class PolynomialKernel {
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> eigenVector;
  typedef Eigen::Matrix<double, 1, Eigen::Dynamic> eigenRowVector;
  typedef Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> eigenIndexVector;
public:
    PolynomialKernel() {}
    PolynomialKernel(unsigned int d_, unsigned int n_) : d(d_), n(n_), basisdim(binomialCoefficient( n_ + d_, d_  )), myIndex(d_,n_),Hm(eigenMatrix::Identity(basisdim,basisdim)) {
    }
  static constexpr double inf = -INFINITY;
  static constexpr double sup = INFINITY;
  void setM(const eigenMatrix& M){
      Hm = M;
  }
  eigenMatrix getM() const {
      return(Hm);
  }
  
  void init(unsigned int d_, unsigned int n_){
      d = d_;
      n = n_;
      basisdim = (binomialCoefficient( n_ + d_, d_  ));
      myIndex.init(d_,n_);
      Hm = eigenMatrix::Identity(basisdim,basisdim);
  }
  
  template <typename Derived, typename otherDerived>
  double operator()(const Eigen::MatrixBase<Derived> &x,
                    const Eigen::MatrixBase<otherDerived> &y) const {
                const auto &mySet = myIndex.get_MultiIndexSet();
                eigenRowVector xx(basisdim);
                eigenVector yy(basisdim);
                double xaccum;
                double yaccum;
                unsigned int counter(0);
                for ( const auto &ind1 : mySet ) {
                    xaccum = 1;
                    yaccum = 1;
                    for(unsigned int j = 0; j < d; ++j){
                        xaccum *= std::pow(x(j),ind1(j));
                        yaccum *= std::pow(y(j),ind1(j));
                    }
                    xx(counter) = xaccum;
                    yy(counter) = yaccum;
                    ++counter;
                }
                return(xx * Hm * yy);

  }
private:
    unsigned int d; //dimension of the variable
    unsigned int n; //order of the variable
    unsigned int basisdim;
    RRCA::MultiIndexSet<eigenIndexVector> myIndex;
    eigenMatrix Hm;
};

} // namespace RRCA

#endif
