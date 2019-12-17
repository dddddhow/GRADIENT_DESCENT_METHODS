#include <iostream>
#include <armadillo>
#include "arma_my_types.h"

using namespace std;
using namespace  arma;

void lasso_func_(void *In_A_raw, void *Out_x_raw, void *In_d_raw, int *Np, float *errr);

int main()
{
    int np=10;
    float err =0.00001;
    cmat  A(np,np,fill::randn);
    cvec x_true(np);
    cvec x_cal(np);
    cvec d(np);

    //随机赋初始值
    mat temp_r = randn(np,np+1);
    mat temp_i = randn(np,np+1);

    //X_true
    for(int i = 0; i < np; i++)
    {
        x_true(i).real(temp_r(i,np));
        x_true(i).imag(temp_i(i,np));
    }

    //x_cal
    x_cal.fill(0.0f);

    //A
    //A.diag() += fcomplex(1.0f,0.0f);
    cmat U,V;
    fvec s;
    svd(U,s,V,A);
    s.print("sgm=");
    s += s.max()*5;
    A = U*diagmat(s)*V.t();

    auto A_ori = A;
    A = A.t()*A;

    //d
    d = A * x_true;


    //GD计算
    lasso_func_(A.memptr() , x_cal.memptr(), d.memptr(), &np ,&err);

    //文件输出
    abs(x_true).print("x_true is :");
    abs(x_cal).print("x_cal is :");


    return 0;
}

void lasso_func_(void *In_A_raw, void *Out_x_raw, void *In_d_raw, int *Np, float *errr)
{
    typedef complex<float> fcomplex;

    complex<float> * In_A  = (complex<float> * )In_A_raw;
    complex<float> * Out_x = (complex<float> * )Out_x_raw;
    complex<float> * In_d  = (complex<float> * )In_d_raw;
    int np=*Np;
    float err=*errr;
    //=============================Definition=================================

    //cmat A(np, np);
    cmat A(In_A, np, np, false, true);
    cvec X(np);
    cvec f(np);
    cvec X1(np), X2(np);
    cvec R(np), R1(np), R2(np);
    cvec P(np), P1(np), P2(np);
    cvec AP(np);
    cvec Ptmp(np);

    double norm2min, errs;
    double alpha   , beta;
    double rs_old  ,rs_new;
    double PAP;

    //memcpy( &A(0,0), In_A_raw, sizeof(fcomplex)*np*np );
    cout<<"real(A)=\n"<<real(A)<<endl;

    for(int i=0;i<np;i++)
    {
        f(i)=In_d[i];
    }

    fvec vc_tmp = {1.0E-2, 5*err};
    norm2min = arma::min(vc_tmp);
    errs     = double(err) * double(err);

    //Zeros
    AP.fill( fcomplex(0.0f)   );
    P.fill( fcomplex(0.0f)   );
    R.fill( fcomplex(0.0f)   );
    X.fill( fcomplex(0.0f)   );

    //Initial
    R = f-A*X;
    P = R;
    rs_new = real(cdot(R,R));
    rs_old = rs_new;

    //==========================Begin CG Process==============================
    for (int iteration = 0; iteration<np; iteration++)
    {

        if(rs_old <= norm2min)
        {
            break;
        }

        //compute alpha dot(r,r)/dot(AP,P)
        AP  = A*P;
        PAP = std::real(cdot(P,AP)) + 1e-3;

        if(abs(PAP) <= norm2min)
        {
            break;
        }
        alpha  = rs_old*1.0/PAP;

        //compute X   X=X+alpha*P
        X      += alpha*P;

        //compute R   R=R-alpha*AP
        R      -= alpha*AP;
        rs_new = real(cdot(R,R));

        //compute Beta Beta = rs_new/(rs_old)
        //beta   = -real(dot_product(R,AP))/(PAP+err*0);
        //beta    = rs_new*1.0/(rs_old+1e-11);
        beta = 0;

        //compute P new
        P       *= beta;
        P       += R;

        if ( iteration % 1 ==0 )
        {
            //print*,'iter,r2norm,r1norm,alpha,beta',iteration,rs_new,rs_old,alpha,beta
        }

        rs_old = rs_new;

        //=========================== L1  Process== ==============================

        float lamda_r ;
        float lamda_i ;
        float temp_r  = abs(X(0).real());
        float temp_i  = abs(X(0).imag());

        for(int i=1; i<np; i++)
        {
            if(temp_r > abs(X(i).real()))
            {
                temp_r = abs(X(i).real());
            }

            if(temp_i > abs(X(i).imag()))
            {
                temp_i = abs(X(i).imag());
            }
        }

        lamda_r = temp_r*1.0;
        lamda_i = temp_i*1.0;

        lamda_r = 0;
        lamda_i = 0;

        cout<<"Lamda_r is "<<lamda_r<<endl;
        cout<<"Lamda_i is "<<lamda_i<<endl;

        for(int i=0; i<np; i++)
        {
            if(X(i).real() == abs(X(i).real()))
            {
                X(i).real(abs(X(i).real()) - lamda_r);
            }
            else if(X(i).real() == -abs(X(i).real()))
            {
                X(i).real(-(abs(X(i).real()) -lamda_r));
            }


            if(X(i).imag() == abs(X(i).imag()))
            {
                X(i).imag(abs(X(i).imag()) - lamda_i);
            }
            else if(X(i).imag() == -abs(X(i).imag()))
            {
                X(i).imag(-(abs(X(i).imag()) -lamda_i));
            }

            if(abs(X(i).real()) < lamda_r)
            {
                X(i).real(0);
            }
            if(abs(X(i).imag()) < lamda_i)
            {
                X(i).imag(0);
            }

        }

    }

    //=============================== Output =================================
    for(int i=0;i<np;i++)
    {
        Out_x[i] =X(i);
    }

}
