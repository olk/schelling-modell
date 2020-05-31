//          Copyright Oliver Kowalke 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cstdlib>

class animation;

namespace schelling {

class model {
private:
    unsigned int * inSrc_ = nullptr;
    unsigned int * outSrc_ = nullptr;
    std::size_t dim_;
    std::size_t states_;

public:
    model( std::size_t dim, std::size_t states);

    ~model();

    void run( animation &);
};

}
