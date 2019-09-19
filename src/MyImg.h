#include<cstdio>
#include<vector>
#define IMG_TYPE float


class MyImg{
/*
 * This self-defined class is a simple substitution for two dimensional numpy.
 * It will be extended with semester going.
 */
public:
    MyImg(int h, int w) {
        this->w = w;
        this->h = h;
        this->img = std::vector<std::vector<IMG_TYPE>>();
        this->img.resize(h);
        for(int i=0; i<h; ++i)
            this->img[i].resize(w);
    };

    void print() {
        for (int i=0; i<this->h; ++i) {
            for (int j=0; j<this->w; ++j)
                printf("%f ", this->img[i][j]);
            printf("\n");
        }
    }

    const int shape(const int& axis) const {
        if (axis == 0)
            return this->h;
        else if (axis == 1)
            return this->w;
        return 0;
    }

    void limit() {
        for (int i=0; i<this->h; ++i)
            for (int j=0; j<this->w; ++j)
                if (this->img[i][j] < 0)
                    this->img[i][j] = 0;
                else if (this->img[i][j] > 1)
                    this->img[i][j] = 1;
    }

    // set
    IMG_TYPE& operator() (const int& x, const int& y) {
        return this->img[x][y];
    }
    // get
    const IMG_TYPE& operator() (const int& x, const int& y) const {
        return this->img[x][y];
    }

    MyImg operator + (const MyImg &img) {
        auto new_img = MyImg(this->w, this->h);
        for (int i=0; i<this->h; ++i)
            for (int j=0; j<this->w; ++j)
                new_img(i, j) = this->img[i][j] + img(i, j);
        return new_img;
    }

    MyImg operator + (const IMG_TYPE &num) {
        auto new_img = MyImg(this->w, this->h);
        for (int i=0; i<this->h; ++i)
            for (int j=0; j<this->w; ++j)
                new_img(i, j) = this->img[i][j] + num;
        return new_img;
    }

    MyImg operator * (const IMG_TYPE &num) {
        auto new_img = MyImg(this->w, this->h);
        for (int i=0; i<this->h; ++i)
            for (int j=0; j<this->w; ++j)
                new_img(i, j) = this->img[i][j] * num;
        return new_img;
    }

    ~MyImg() {}

private:
    int w, h;
    std::vector<std::vector<IMG_TYPE>> img;
};
