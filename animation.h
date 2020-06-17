#ifndef ANIMATION_H
#define ANIMATION_H

#include <cstdlib>
#include <functional>

#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>

class animation {
private:
    unsigned char                       *   pixels_ = nullptr;
    std::size_t                             x;
    std::size_t                             y;
    std::function< void(unsigned int) >     anim_fn;

    // static method used for glut callbacks
    static animation ** get_animation_ptr() {
        static animation * ganimation;
        return & ganimation;
    }

    // static method used for glut callbacks
    static void idle_fn() {
        static unsigned int ticks = 1;
        animation * animation = * ( get_animation_ptr() );
        animation->anim_fn( ticks++);
        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void Key( unsigned char key, int, int) {
        switch ( key) {
            case 27:
                std::exit( 0);
        }
    }

    // static method used for glut callbacks
    static void Draw() {
        animation * animation = * ( get_animation_ptr() );
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT);
        glDrawPixels( animation->x, animation->y, GL_RGBA, GL_UNSIGNED_BYTE, animation->pixels_);
        glutSwapBuffers();
    }

public:
    animation( std::size_t width, std::size_t height);

    ~animation();

    unsigned char * get_ptr() const noexcept {
        return pixels_;
    }

    long size() const noexcept {
        return x * y * 4;
    }

    template< typename Fn >
    void display_and_exit( Fn && fn) {
        animation ** animation = get_animation_ptr();
        * animation = this;
        anim_fn = fn;
        int c = 1;
        char * dummy = nullptr;
        glutInit( & c, & dummy );
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowSize( x, y);
        glutCreateWindow("animation");
        glutKeyboardFunc( Key);
        glutDisplayFunc( Draw);
        glutIdleFunc( idle_fn);
        glutMainLoop();
    }
};

#endif  // ANIMATION_H
