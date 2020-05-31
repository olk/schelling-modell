//          Copyright Oliver Kowalke 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "model.h"

#include <cmath>
#include <random>
#include <tuple>

#include <cuda.h>

#include "animation.h"

__global__
void cell( std::size_t dim, std::size_t states, unsigned int * in, unsigned int * out) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ( x < dim && y < dim) {
        int offset = x + y * dim;
        unsigned int t = in[x+(y+1)%dim*dim];
        unsigned int l = in[(0 == x) ? dim-1+y*dim : (x-1)%dim+y*dim];
        unsigned int c = in[offset];
        unsigned int r = in[(x+1)%dim+y*dim];
        unsigned int b = in[(0 == y) ? x+(dim-1)*dim : x+(y-1)%dim*dim];
        unsigned int tm = t % states;
        unsigned int lm = l % states;
        unsigned int cm = (c + 1) % states;
        unsigned int rm = r % states;
        unsigned int bm = b % states;
        if ( tm == cm) {
            out[offset] = t;
        } else if ( lm == cm) {
            out[offset] = l;
        } else if ( rm == cm) {
            out[offset] = r;
        } else if ( bm == cm) {
            out[offset] = b;
        }
    }
}

__device__
std::tuple< unsigned char, unsigned char, unsigned char >
rgb_from_int( std::size_t states, unsigned int state) {
    float f = static_cast< float >(state)/(states-1);
    return std::make_tuple(
            static_cast< unsigned char >(255*f),
            static_cast< unsigned char >(0),
            static_cast< unsigned char >(255*(1-f)));
}

__global__
void map( std::size_t dim, std::size_t states, unsigned char* optr, unsigned int * inSrc, unsigned int const* outSrc) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ( x < dim && y < dim) {
        int offset = x + y * dim;
        int state = outSrc[offset];
        inSrc[offset] = state;
        auto t = rgb_from_int( states, state);
        optr[4 * offset + 0] = std::get<0>( t);
        optr[4 * offset + 1] = std::get<1>( t);
        optr[4 * offset + 2] = std::get<2>( t);
        optr[4 * offset + 3] = 255;
    }
}

namespace schelling {

model::model( std::size_t dim, std::size_t states) :
    dim_{ dim }, states_{ states } {
    // initialize memory
    std::size_t size = dim_ * dim_ * sizeof( unsigned int);
    cudaMallocManaged( & inSrc_, size);
    cudaMallocManaged( & outSrc_, size);
    // random initialization of cells
    std::minstd_rand generator;
    std::uniform_int_distribution<> distribution{ 0, states_-1 };
    for ( unsigned int i = 0; i < dim_ * dim_; ++i) {
        inSrc_[i] = distribution( generator);
        outSrc_[i] = inSrc_[i];
    }
}

model::~model() {
    // release memory
    cudaFree( inSrc_);
    cudaFree( outSrc_);
}

void
model::run( animation & image) {
    const std::size_t x = std::ceil( dim_/32.0);
    const dim3 dim_grid{ x, x };
    const dim3 dim_block{ 32, 32 };
    image.display_and_exit(
        [dim_grid,dim_block,&image,this](unsigned int) mutable {
            cell<<< dim_grid, dim_block >>>( dim_, states_, inSrc_, outSrc_);
            map<<< dim_grid, dim_block >>>( dim_, states_, image.get_ptr(), inSrc_, outSrc_);
            cudaDeviceSynchronize();
        });
}

}
