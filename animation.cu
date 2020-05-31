#include "animation.h"

#include <cuda.h>

animation::animation( std::size_t width, std::size_t height) :
        x( width),
        y( height) {
    cudaMallocManaged( & pixels_, size() );
}

animation::~animation() {
    cudaFree( pixels_);
}
