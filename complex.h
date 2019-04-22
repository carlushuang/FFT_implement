#ifndef __COMPLEX_H
#define __COMPLEX_H

#include <iostream>
#include <cmath>

#ifndef C_PI
#define C_PI  3.14159265358979323846
#define C_2PI 6.28318530717958647692
#endif

template<typename T>
class complex_t{
private:
    T im_data;
    T re_data;
public:
    typedef complex_t<T> c_type;
    complex_t(){
        im_data = 0;
        re_data = 0;
    }
    complex_t(T re_, T im_){
        re_data = re_;
        im_data = im_;
    }
    complex_t(const complex_t<T> & rhs){
        im_data = rhs.im();
        re_data = rhs.re();
    }
    T & im(){return im_data;}
    const T & im() const {return im_data;}
    T & re(){return re_data;}
    const T & re() const {return re_data;}

    complex_t<T> & operator=(const complex_t<T>& rhs){
        if(this != &rhs){
            im_data = rhs.im();
            re_data = rhs.re();
        }
        return *this;
    }

    complex_t<T> & operator *= (const complex_t<T>& rhs){
        T im_ = this->im();
        T re_ = this->re();
        this->re() = re_ * rhs.re() - im_ * rhs.im();
        this->im() = re_ * rhs.im() + im_ * rhs.re();
        return *this;
    }
    complex_t<T> operator *(const complex_t<T>& rhs){
        complex_t<T> result(this->re(), this->im());
        result *= rhs;
        return result;
    }
    complex_t<T> operator *(const complex_t<T>& rhs) const {
        return const_cast<c_type*>(this)->operator *(rhs);
    }

    complex_t<T> & operator *=(const T & scalar){
        this->re() = this->re() * scalar;
        this->im() = this->im() * scalar;
        return *this;
    }
    complex_t<T> operator *(const T & scalar){
        complex_t<T> result(this->re(), this->im());
        result *= scalar;
        return result;
    }
    complex_t<T> operator *(const T & scalar)const{
        return const_cast<c_type*>(this)->operator *(scalar);
    }

    complex_t<T> & operator /=(const complex_t<T>& rhs){
        /*
         *   a+bi   a+bi   c-di   ac+bd + i(bc-ad)
         *   ---- = ---- * ---- = ----------------
         *   c+di   c+di   c-di      c^2 + d^2
         */
        T a = this->re();
        T b = this->im();
        T c = rhs.re();
        T d = rhs.im();

        T l = c*c + d*d;
        this->re() = (a*c+b*d)/l;
        this->im() = (b*c-a*d)/l;
        return *this;
    }
    complex_t<T> operator /(const complex_t<T>& rhs){
        complex_t<T> result(this->re(), this->im());
        result /= rhs;
        return result;
    }
    complex_t<T> operator /(const complex_t<T>& rhs)const{
        return const_cast<c_type*>(this)->operator/(rhs);
    }

    complex_t<T> & operator /=(const T & scalar){
        this->re() = this->re() / scalar;
        this->im() = this->im() / scalar;
        return *this;
    }
    complex_t<T>  operator /(const T & scalar){
        complex_t<T> result(this->re(), this->im());
        result /= scalar;
        return result;
    }
    complex_t<T> operator /(const T & scalar)const{
        return const_cast<c_type*>(this)->operator/(scalar);
    }

    complex_t<T> & operator +=(const complex_t<T>& rhs){
        T im_ = this->im();
        T re_ = this->re();

        this->re() = re_ + rhs.re();
        this->im() = im_ + rhs.im();
        return *this;
    }
    complex_t<T> operator+(const complex_t<T>& rhs){
        complex_t<T> result(this->re(), this->im());
        result += rhs;
        return result;
    }
    complex_t<T> operator+(const complex_t<T>& rhs)const{
        return const_cast<c_type*>(this)->operator+(rhs);
    }

    complex_t<T> & operator -= (const complex_t<T>& rhs){
        T im_ = this->im();
        T re_ = this->re();
        this->re() = re_ - rhs.re();
        this->im() = im_ - rhs.im();
        return *this;
    }
    complex_t<T> operator -(const complex_t<T>& rhs){
        complex_t<T> result(this->re(), this->im());
        result -= rhs;
        return result;
    }
    complex_t<T> operator-(const complex_t<T>& rhs)const{
        return const_cast<c_type*>(this)->operator-(rhs);
    }
};
template<typename T>
std::ostream & operator<<(std::ostream & t, const complex_t<T> & c){
    t<<c.re()<<"+"<<c.im()<<"j";
    return t;
}
namespace std{
    template<typename T>
    T real(const complex_t<T> & c){
        return c.re();
    }

    template<typename T>
    T imag(const complex_t<T> & c){
        return c.im();
    }

    template<typename T>
    T abs(const complex_t<T> & c){
        return std::sqrt(c.re()*c.re()+c.im()*c.im());
    }

    template<typename T>
    complex_t<T> conj(const complex_t<T> & c){
        complex_t<T> result(c.re(), ((T)-1) * c.im());
        return result;
    }

    template<typename T>
    T arg(const complex_t<T> & c){
        // std::atan2(y, x), (-pi ~ pi)
        return std::atan2(c.im(), c.re());
    }

    // std::polar can not distinguish
    template< class T > 
    complex_t<T> polar2( const T& r, const T& theta = T()){
        // r*e^(i*theta) = r*(cos(theta) + i* sin(theta))
        T re = r*std::cos(theta);
        T im = r*std::sin(theta);
        return complex_t<T>(re, im);
    }

    // compute x^y
    template<typename T>
    complex_t<T> pow(const complex_t<T> & x, const complex_t<T> & y){
        // https://en.wikipedia.org/wiki/Exponentiation#Computing_complex_powers
        // x = r*e^(i*theta)    polar form
        // y = c+di             Cartesian form
        // x^y = (r^c)*e^(-d*theta) * e^(i*(d*log(r)+c*theta)), polar form
        T r = std::abs(x);
        T theta = std::arg(x);
        T c = y.re();
        T d = y.im();

        T result_r = std::pow(r, c) * std::exp(-1*d*theta);
        T result_theta = d*std::log(r) + c*theta;

        complex_t<T> result = std::polar2(result_r, result_theta);
        return result;
    }
    template<typename T>
    complex_t<T> pow(const complex_t<T> & x, const T & y){
        complex_t<T> yy(y, 0);
        return std::pow(x, yy);
    }
    template<typename T>
    complex_t<T> pow(const T & x, const complex_t<T> y){
        // b^z = e^(z*log(b)) = e^( (r*cos(theta) + r*sin(theta)*i) *log(b))
        //     = e^(r*cos(theta)*log(b)) * e^( r*log(b)*sin(theta)*i )
        T r = std::abs(y);
        T theta = std::arg(y);
        T result_r = std::exp( r*std::cos(theta)*std::log(x) );
        T result_theta = r*std::log(x)*std::sin(theta);
        return std::polar2(result_r, result_theta);
    }

    template<typename T>
    complex_t<T> exp(const complex_t<T> & c){
        // e^z = e^( r*cos(theta)+r*sin(theta)*i )
        //    = e^(r*cos(theta)) * e^(r*sin(theta))
        T r = std::abs(c);
        T theta = std::arg(c);

        T result_r = std::exp(r * std::cos(theta));
        T result_theta = r * std::sin(theta);
        return std::polar2(result_r, result_theta);
    }
}

typedef complex_t<float>    complex_fp32_t;
typedef complex_t<double>   complex_fp64_t;
#if 0
    // test
    complex_fp32_t c1(1.04,-3.12789);
    std::complex<float> c1_(1.04, -3.12789);
    complex_fp32_t c2(-0.721,0.2);
    std::complex<float> c2_(-0.721,0.2);

    std::cout<<std::pow(c1, (float)0.32)<<std::endl;
    std::cout<<std::pow(c1_, (float)0.32)<<std::endl;
    std::cout<<"---------------"<<std::endl;
    std::cout<<std::pow(c1, c2)<<std::endl;
    std::cout<<std::pow(c1_, c2_)<<std::endl;
    std::cout<<"---------------"<<std::endl;
    std::cout<<std::pow((float)2.134, c2)<<std::endl;
    std::cout<<std::pow((float)2.134, c2_)<<std::endl;
    std::cout<<"---------------"<<std::endl;
    std::cout<<std::exp(c2)<<std::endl;
    std::cout<<std::exp( c2_)<<std::endl;
#endif

#endif