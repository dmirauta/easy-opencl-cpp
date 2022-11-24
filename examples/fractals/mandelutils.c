#include "mandelstructs.h"

Complex_t complex_add(Complex_t a, Complex_t b)
{
    Complex_t c;
    c.re = a.re + b.re;
    c.im = a.im + b.im;
    return c;
}

Complex_t complex_mult(Complex_t a, Complex_t b)
{
    Complex_t c;
    c.re = a.re*b.re - a.im*b.im;
    c.im = a.im*b.re + a.re*b.im;
    return c;
}

// function which we recurse
Complex_t f(Complex_t z, Complex_t c)
{
    return complex_add(complex_mult(z, z), c);
}

int in_circle(Complex_t z, Complex_t z0, FPN r)
{
    FPN dre = z.re - z0.re;
    FPN dim = z.im - z0.im;
    return dre*dre + dim*dim < r*r;
}

int in_box(Complex_t z, Box_t b)
{
    return z.re > b.left && z.re<b.right && z.im>b.bot && z.im<b.top;
}

int in_bounds(Complex_t z)
{
    return in_circle(z, (Complex_t){0.0,0.0}, 2);
}

FPN _abs(FPN x)
{
    if(x>0)
    {
        return x;
    } else {
        return -x;
    }
}

FPN _min(FPN a, FPN b)
{
    if(a<b)
    {
        return a;
    } else {
        return b;
    }
}

FPN proximity(Complex_t z, int PROXTYPE)
// Various things we can measure distance from...
{
  FPN res = 1000.0;
  if (PROXTYPE & 1)
  {
    res = _min(res, z.re*z.re + z.im*z.im);
  }
  if (PROXTYPE & 2)
  {
    res = _min(res, _abs(z.re));
  }
  if (PROXTYPE & 4)
  {
    res = _min(res, _abs(z.im));
  }
  return res;
}

int _escape_iter(Complex_t z, Complex_t c, int MAXITER)
{

    int i=0;
    while(i<MAXITER && in_bounds(z))
    {
        z = f(z, c);
        i+=1;
    }

    return i;
}

FPN _minprox(Complex_t z, Complex_t c, int MAXITER, int PROXTYPE)
// more of a distance field?
{

    int i=0;
    FPN dist = proximity(z, PROXTYPE);
    while(i<MAXITER)
    {
        z = f(z, c);
        dist = _min(dist, proximity(z, PROXTYPE));
        i+=1;
    }

    return dist;
}

Complex_t _orbit_trap(Complex_t z, Complex_t c, Box_t b, int MAXITER)
// returns UV coords in given box
{
    Complex_t res = {-b.left, -b.bot};

    int i=0;
    while(i<MAXITER)
    {
        i+=1;
        z = f(z, c);
        if(in_box(z, b))
        {
          res = complex_add(res, z);
          res.re/=(b.right-b.left);
          res.im/=(b.top-b.bot);
          return res;
        }

    }

    return (Complex_t){0.0, 0.0};
}
