#include <iostream>
#include <stdint.h>
#include "Rasterizer.h"

void init_points(double* points) 
{
    //v1
    points[0]  =  0.f;
    points[1]  =  0.f;
    points[2]  =  0.f;
    points[3]  =  1.f;
    //v2
    points[4]  =  0.f;
    points[5]  =  0.75f;
    points[6]  =  0.f;
    points[7]  =  1.f;
    //v3
    points[8]  =  0.75f;
    points[9]  =  0.f;
    points[10] =  0.f;
    points[11] =  1.f;
}

void set_pixel_values(uint8_t *p, uint8_t R, uint8_t G, uint8_t B, uint8_t A)
{
    *p++ = R;
    *p++ = G;
    *p++ = B;
    *p++ = A;
}

void init_texture(uint8_t *texture, unsigned n_pixels) 
{
    unsigned i = 0;
    for(; i < n_pixels; i += 4)
        set_pixel_values(&texture[i], 255, 0, 0, 255);
    for(; i < n_pixels * 2; i += 4)
        set_pixel_values(&texture[i], 0, 255, 0, 255);
    for(; i < n_pixels * 3; i += 4)
        set_pixel_values(&texture[i], 0, 0, 255, 255);
    for(; i < n_pixels * 4; i += 4)
        set_pixel_values(&texture[i], 255, 255, 0, 255);
}

void init_tcoords(float* tc)
{
    tc[0]  = 0.;
    tc[1]  = 0.;

    tc[2]  = 0.;
    tc[3]  = 1.;

    tc[4]  = 1.;
    tc[5]  = 0.;
}

void init_color(float* c)
{
    c[0]  = 0.;
    c[1]  = 0.;
    c[2]  = 1.;

    c[3]  = 0.;
    c[4]  = 0.;
    c[5]  = 1.;

    c[6]  = 0.;
    c[7]  = 0.;
    c[8]  = 1.;
}

int main(int argc, char** argv)
{
    unsigned int trilist[] =
    {
        0, 2, 1,
    };
    size_t n_points = 3;
    size_t n_tris = 1;
    size_t t_w = 64;
    size_t t_h = 64;
    double *points = new double[n_points * 4];
    float *color = new float[n_points * 3];
    uint8_t *texture = new uint8_t[t_w * t_h * 4];
    float *tcoords = new float[n_points * 2];

    init_points(points);
    init_tcoords(tcoords);
    init_texture(texture, t_w * t_h);
    init_color(color);

    Rasterizer rasterizer(points, color, n_points, trilist, n_tris, tcoords, 
            texture, t_w, t_h, true);

    int output_w = 128;
    int output_h = 128;
    uint8_t* pixels = new uint8_t[output_w * output_h * 4];
    float* colorResult = new float[output_w * output_h * 3];
    rasterizer.render(argc, argv);

    int count = 0;
    for(int i = 0; i < output_h * output_w * 3; i++) {
        if(colorResult[i] > 0.1)
            count++;
    }
    double ratio  = count * 1.0 / (output_h * output_w * 3.0);
    std::cout << "Proportion non-black: " << ratio << std::endl;
    delete[] points;
    delete[] color;
    delete[] texture;
    delete[] tcoords;
    delete[] pixels;
    delete[] colorResult;
    return(EXIT_SUCCESS);
}

