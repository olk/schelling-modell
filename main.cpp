//          Copyright Oliver Kowalke 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "animation.h"
#include "modell.h"

int main() {
    std::size_t dim = 2048, states = 15;
    schelling::model model{ dim, states };
    animation image{ dim, dim };
    model.run( image);
    return EXIT_SUCCESS;
}
