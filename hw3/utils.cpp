/*
 * Title: Some utilities function speed up by cpp written here.
 * Author: linnil1
 */
#include "MyImg.h"
#include <functional>


// get median from histogram
int histMed(int* hist, int size) {
    // Calculate the location of median
    int aim[2];
    if (size % 2)
        aim[0] = size / 2 + 1;
    else
        aim[0] = size / 2;
    aim[1] = size / 2 + 1;

    // Summation the histogram
    int r[2], j=0;
    for(int i=0, s=0; i<256; ++i) {
        s += hist[i];
        if (s >= aim[j]) {
            r[j] = i;
            ++j;
        }
        if (j < 2 && s >= aim[j]) {
            r[j] = i;
            ++j;
        }
        // output the median
        if (j == 2)
            return (r[0] + r[1]) / 2;
    }
    // Error! Output 0
    return 0;
}


// get min from histogram
int histMin(int* hist, int size) {
    for(int i=0; i<256; ++i)
        if (hist[i])
            return i;
    return 0;
}


// get max from histogram
int histMax(int* hist, int size) {
    for(int i=255; i>=0; --i)
        if (hist[i])
            return i;
    return 0;
}


// Wrap ordered operation into one function
// For median, min, max filter
// Note oridata is padding image, (nx, ny) are original image, (sizex, sizey) are kernal size
MyImg<float> orderFilter(MyImg<float> &oridata, int nx, int ny, int sizex, int sizey, std::function<int(int*, int)> func) {
    MyImg<int> data = oridata.toInt();
    MyImg<float> new_img(nx, ny);
    int s = sizex * sizey;

    for(int i=0; i<nx; ++i) {
        int hist[256]{0};
        for(int x=0; x<sizex; ++x)
            for(int y=0; y<sizey; ++y)
                hist[data(i+x, y)] += 1;
        new_img(i, 0) = func(hist, s);
        for(int j=1; j<ny; ++j) {
            for(int x=0; x<sizex; ++x) {
                hist[data(i+x, j-1)] -= 1;
                hist[data(i+x, j-1+sizey)] += 1;
            }
            new_img(i, j) = func(hist, s);
        }
    }
    new_img.toFloat();
    return new_img;
}


// wrap
MyImg<float> medianFilter(MyImg<float> &oridata, int nx, int ny, int sizex, int sizey) {
    return orderFilter(oridata, nx, ny, sizex, sizey, histMed);
}


// wrap
MyImg<float> minFilter(MyImg<float> &oridata, int nx, int ny, int sizex, int sizey) {
    return orderFilter(oridata, nx, ny, sizex, sizey, histMin);
}


// wrap
MyImg<float> maxFilter(MyImg<float> &oridata, int nx, int ny, int sizex, int sizey) {
    return orderFilter(oridata, nx, ny, sizex, sizey, histMax);
}

// test
int main() {
    int nx=10, ny=20,
        padnx=14, padny=24,
        sizex=5, sizey=5;
    MyImg<float> data(padnx, padny);
    for(int i=0; i<padnx; ++i)
        for(auto j=0; j<padny; ++j)
            data(i, j) = (i * padny + j) % 256;
    data.toFloat();

    printf("12345\n");
    auto new_img = medianFilter(data, nx, ny, sizex, sizey);
    printf("12345\n");
    new_img.print();
    printf("12345\n");
}
