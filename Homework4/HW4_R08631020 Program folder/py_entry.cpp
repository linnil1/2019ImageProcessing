/*
 * Title: This file is the bridge of my cpp file and python object
 * Author: linnil1
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "MyImg.h"


PyObject * medianFilterPy(PyObject*, PyObject*);
PyObject * minFilterPy(PyObject*, PyObject*);
PyObject * maxFilterPy(PyObject*, PyObject*);
MyImg<float> medianFilter(MyImg<float>&, int, int, int, int);
MyImg<float> minFilter(MyImg<float>&, int, int, int, int);
MyImg<float> maxFilter(MyImg<float>&, int, int, int, int);


// define methods for module
static PyMethodDef Methods[] = {
    {"medianFilter", medianFilterPy, METH_VARARGS, "Median Filter"},
    {"minFilter", minFilterPy, METH_VARARGS, "Max Filter"},
    {"maxFilter", maxFilterPy, METH_VARARGS, "Min Filter"},
    {NULL},
};


// Define a module
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "utils_cpp", NULL, -1, Methods
};


// Create a module
PyMODINIT_FUNC PyInit_utils_cpp(void)
{
    return PyModule_Create(&module);
}


// PyObject to MyImg
MyImg<float> parseToMyImg(PyObject* obj) {
    Py_ssize_t h = PyList_Size(obj),
               w = PyList_Size(PyList_GetItem(obj, 0));
    MyImg<float> img{int(h), int(w)};
    for (int i=0; i<h; ++i)
        for (int j=0; j<w; ++j)
            img(i, j) = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(obj, i), j));
    return img;
}


// MyImg to PyObject
PyObject* parseToObj(const MyImg<float>& img) {
    PyObject* obj = PyList_New(img.shape(0));
    for (int i=0; i<img.shape(0); ++i) {
        auto row_obj = PyList_New(img.shape(1));
        for (int j=0; j<img.shape(1); ++j)
            PyList_SetItem(row_obj, j,  PyFloat_FromDouble(img(i, j)));
        PyList_SetItem(obj, i, row_obj);
    }
    return obj;
}


// Below three functions wrap functions in utils.cpp to python obj
PyObject* medianFilterPy(PyObject *self, PyObject *args) {
    // read args
    PyObject *image_obj;
    int sx, sy, nx, ny;
    if (!PyArg_ParseTuple(args, "O!iiii", &PyList_Type, &image_obj, &nx, &ny, &sx, &sy))
        return NULL;

    // main
    MyImg<float> img = parseToMyImg(image_obj);
    // printf("%d %d %d %d\n", img.shape(0), img.shape(1), nx, ny);
    auto new_img = medianFilter(img, nx, ny, sx, sy);
    return parseToObj(new_img);
}


PyObject* minFilterPy(PyObject *self, PyObject *args) {
    // read args
    PyObject *image_obj;
    int sx, sy, nx, ny;
    if (!PyArg_ParseTuple(args, "O!iiii", &PyList_Type, &image_obj, &nx, &ny, &sx, &sy))
        return NULL;

    // main
    MyImg<float> img = parseToMyImg(image_obj);
    // printf("%d %d %d %d\n", img.shape(0), img.shape(1), nx, ny);
    auto new_img = minFilter(img, nx, ny, sx, sy);
    return parseToObj(new_img);
}


PyObject* maxFilterPy(PyObject *self, PyObject *args) {
    // read args
    PyObject *image_obj;
    int sx, sy, nx, ny;
    if (!PyArg_ParseTuple(args, "O!iiii", &PyList_Type, &image_obj, &nx, &ny, &sx, &sy))
        return NULL;

    // main
    MyImg<float> img = parseToMyImg(image_obj);
    // printf("%d %d %d %d\n", img.shape(0), img.shape(1), nx, ny);
    auto new_img = maxFilter(img, nx, ny, sx, sy);
    return parseToObj(new_img);
}
