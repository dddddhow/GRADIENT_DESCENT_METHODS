/***************************************
*     Author:Xu Peng
*     Date:2019-03-08 20:30
*     Filename:arma_my_types.h
*     Description:
*
*     Last modified:2019-03-08 20:30
****************************************/
#pragma once
#include <armadillo>
#include <complex>

typedef  std::complex<float> fcomplex;
using namespace arma;
typedef Mat<fcomplex>  cmat;
typedef Row<fcomplex>  crow;
typedef Col<fcomplex>  ccol;
typedef Col<fcomplex>  cvec;
typedef Cube<fcomplex> ccube;

//typedef Mat<float>    fmat;
typedef Row<float>    frow;
typedef Col<float>    fcol;
//typedef Col<float>    fvec;
//typedef Cube<float>   fcube;

typedef Row<sword>    irow;
typedef Col<sword>    icol;
//typedef Col<sword>    ivec;
//typedef Mat<sword>    imat;
//typedef Cube<sword>   icube;

/* INT */
typedef Row<int>    introw;
typedef Col<int>    intcol;
typedef Col<int>    intvec;
typedef Mat<int>    intmat;
typedef Cube<int>   intcube;

//typedef Col<uword>    ucol;
//typedef Row<uword>    urow;
//typedef Col<uword>    uvec;
//typedef Mat<uword>    umat;
//typedef Cube<uword>   ucube;

//typedef fvec::fixed<3> fvec3;
//typedef fvec::fixed<2> fvec2;
typedef cvec::fixed<3> cvec3;
typedef cvec::fixed<2> cvec2;

typedef intvec::fixed<3> intvec3;
typedef intvec::fixed<2> intvec2;


