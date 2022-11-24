#include "mandelutils.c"

__kernel void escape_iter(__global int *res_g,
                          __global EIParam_t *param)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param->view_rect.left + j*(param->view_rect.right-param->view_rect.left)/M,
                   param->view_rect.bot  + i*(param->view_rect.top  -param->view_rect.bot )/N};

    Complex_t _c = param->mandel ? p : param->c;

    res_g[i*M+j] = _escape_iter(p, _c, param->MAXITER);
}

__kernel void min_prox(__global FPN *res_g,

                        // General fract iter params
                        int           mandel, // mandel or julia
                        Complex_t     c,      // not given when mandel selected
                        Box_t         view_rect,

                        // min prox param
                        int           MAXITER,
                        int           PROXTYPE)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {view_rect.left + j*(view_rect.right-view_rect.left)/M,
                   view_rect.bot  + i*(view_rect.top  -view_rect.bot )/N};

    Complex_t _c = mandel ? p : c;

    res_g[i*M+j] = _minprox(p, _c, MAXITER, PROXTYPE);
}

__kernel void orbit_trap(__global Complex_t *res_g,

                                  // General fract iter params
                                  int        mandel, // mandel or julia
                                  Complex_t  c,      // not given when mandel selected
                                  Box_t      view_rect,

                                  // orbit trap param
                                  Box_t      trap,
                                  int        MAXITER)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {view_rect.left + j*(view_rect.right-view_rect.left)/M,
                   view_rect.bot  + i*(view_rect.top  -view_rect.bot )/N};

    Complex_t _c = mandel ? p : c;

    res_g[i*M+j] = _orbit_trap(p, _c, trap, MAXITER);
}

__kernel void map_img   (__global Complex_t *res_g, // result of orbit trap
                         __global Pixel_t   *sim_g, // sample image
                         __global Pixel_t   *mim_g, // mapped image
                                  int        imH,
                                  int        imW)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    int _i = (int) ( ((float) (imH-1)) * res_g[i*M+j].im );
    int _j = (int) ( ((float) (imW-1)) * res_g[i*M+j].re );

    mim_g[i*M+j] = sim_g[_i*imW + _j];

}
