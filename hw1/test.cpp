#include "MyImg.h"

// test function fo MyImg.h
int main() {
    MyImg img{3, 3};
    img(1, 1) = .3;
    MyImg n = (img + .1) * 2 + img;
    n.limit();
    n.print();
    return 0;
}
