/*
 * Title: Some utilities function speed up by cpp written here.
 * Author: linnil1
 */
#include "MyImg.h"
#include <functional>
#include <complex>
#include <vector>
#include <iostream>
#include <cmath>

#define cmp std::complex<double>
#define vcmp std::vector<cmp>
using namespace std::complex_literals;


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


// fft for 1D (Not connect to python)
// Should be 2power size
void fft(vcmp &data, vcmp &new_data, int start, int step) {
    if (start + step >= data.size()) {
        new_data[start] = data[start];
        return;
    }
    fft(data, new_data, start       , step * 2);
    fft(data, new_data, start + step, step * 2);

    int n = data.size(),
        htime = ((n - start - 1) / step + 1) / 2;
    cmp tmp[htime];
    for (int i=start, r=0; i<n; i+=step*2, ++r) {
        cmp e = exp(-M_PI * 2i * cmp(r) / cmp(n)),
            even = new_data[i],
            odd  = new_data[i + step]; 
        new_data[start + step * r] = even + odd * e;
        tmp[r] = even - odd * e;
    }
    for (int i=start + step * htime, r=0; i<n; i+=step, ++r)
        new_data[i] = tmp[r];
}


// fft for 2D (Not connect to python)
// Should be 2power size
void fft2d(MyImg<float> &v1, MyImg<float> &v2) {
    int h = v1.shape(0),
        w = v1.shape(1);
    std::vector<vcmp> new_data;
    for (int i=0; i<h; ++i) {
        vcmp tmp(w);
        for (int j=0; j<w; ++j)
            tmp[j] = cmp(v1(i, j)) + cmp(v2(i, j)) * 1i;
        fft(tmp, tmp, 0, 1);
        new_data.push_back(tmp);
    }

    for (int j=0; j<w; ++j) {
        vcmp tmp(h);
        for (int i=0; i<h; ++i)
            tmp[i] = new_data[i][j];
        fft(tmp, tmp, 0, 1);
        for (int i=0; i<h; ++i) {
            // new_data[i][j] = tmp[i];
            v1(i, j) = real(tmp[i]);
            v2(i, j) = imag(tmp[i]);
        }
    }
}

// test
int main() {
    MyImg<float> v1(4, 4);
    MyImg<float> v2(4, 4);
    v1(0,0)=1; v1(0,1)=2; v1(0,2)=42; v1(0,3)=3;
    v1(1,0)=1; v1(1,1)=12; v1(1,2)=4; v1(1,3)=3;
    v1(2,0)=51; v1(2,1)=2; v1(2,2)=44; v1(2,3)=3;
    v1(3,0)=1; v1(3,1)=2; v1(3,2)=4; v1(3,3)=34;
    fft2d(v1, v2);
    v1.print();
    v2.print();
    /*
    MyImg<float> v1(128, 128);
    MyImg<float> v2(128, 128);
    for(int i=0; i<128; ++i)
        for(int j=0; j<128; ++j)
            v1(i, j) = i + j;
    for(int i=0; i<100; ++i)
        fft2d(v1, v2);
    */

    /*
    vcmp data{1, 2, 4, 4};
    vcmp new_data(4);
    fft(data, new_data, 0, 1);
    for(auto i:new_data)
        std::cout << i << " ";
    std::cout << std::endl;
    */
    /*
    int nx=500, ny=500,
        padnx=550, padny=550,
        sizex=11, sizey=11;
    MyImg<float> data(padnx, padny);
    for(int i=0; i<padnx; ++i)
        for(auto j=0; j<padny; ++j)
            data(i, j) = (i * padny + j) % 256;
    data.toFloat();

    MyImg<float> kernal(10, 10);
    kernal(1, 0) = kernal(0, 1) = kernal(1, 2) = kernal(2, 1) = -1;
    kernal(1, 1) = 5;

    auto new_img = conv(data, nx, ny, kernal);
    for(int i=0; i<9; ++i)
        new_img = conv(data, nx, ny, kernal);
    new_img.print();
    */
}
