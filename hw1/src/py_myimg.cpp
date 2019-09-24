#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "MyImg.h"

PyObject * imageAdd(PyObject*, PyObject*);
PyObject * imageMult(PyObject*, PyObject*);
PyObject * imageAvg(PyObject*, PyObject*);
PyObject * imageSpecial(PyObject*, PyObject*);

// define methods for module
static PyMethodDef Methods[] = {
    {"imageAdd",            imageAdd,     METH_VARARGS, "Add a const number"},
    {"imageMult",           imageMult,    METH_VARARGS, "Multiply a const number"},
    {"imageAvg",            imageAvg,     METH_VARARGS, "Average of two images"},
    {"image_special_func",  imageSpecial, METH_VARARGS, "Special function"},
    {NULL},
};

// Define a module
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "MyImg", NULL, -1, Methods
};

// Create a module
PyMODINIT_FUNC PyInit_myimg(void)
{
    return PyModule_Create(&module);
}

// PyObject to MyImg
MyImg parseToMyImg(PyObject* obj) {
    Py_ssize_t h = PyList_Size(obj),
               w = PyList_Size(PyList_GetItem(obj, 0));
    MyImg img{int(h), int(w)};
    for (int i=0; i<h; ++i)
        for (int j=0; j<w; ++j)
            img(i, j) = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(obj, i), j));
    return img;
}


// MyImg to PyObject
PyObject* parseToObj(const MyImg& img) {
    PyObject* obj = PyList_New(img.shape(0));
    for (int i=0; i<img.shape(0); ++i) {
        auto row_obj = PyList_New(img.shape(1));
        for (int j=0; j<img.shape(1); ++j)
            PyList_SetItem(row_obj, j,  PyFloat_FromDouble(img(i, j)));
        PyList_SetItem(obj, i, row_obj);
    }
    return obj;
}


PyObject* imageAdd(PyObject *self, PyObject *args) {
    // read args
    PyObject *image_obj;
    float num;
    if (!PyArg_ParseTuple(args, "O!f", &PyList_Type, &image_obj, &num))
        return NULL;

    // main
    auto img = parseToMyImg(image_obj);
    img = img + num / 256;
    img.limit();
    return parseToObj(img);
}


PyObject* imageMult(PyObject *self, PyObject *args) {
    // read args
    PyObject *image_obj;
    float num;
    if (!PyArg_ParseTuple(args, "O!f", &PyList_Type, &image_obj, &num))
        return NULL;

    // main
    auto img = parseToMyImg(image_obj);
    img = img * num;
    img.limit();
    return parseToObj(img);
}


PyObject* imageAvg(PyObject *self, PyObject *args) {
    // read args
    PyObject *image_obj1, *image_obj2;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &image_obj1, &PyList_Type, &image_obj2))
        return NULL;

    // main
    auto img1 = parseToMyImg(image_obj1),
         img2 = parseToMyImg(image_obj2);
    MyImg img = (img1 + img2) * .5;
    img.limit();
    return parseToObj(img);
}


PyObject* imageSpecial(PyObject *self, PyObject *args) {
    // read args
    PyObject *image_obj;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &image_obj))
        return NULL;

    // main
    MyImg img = parseToMyImg(image_obj),
          new_img{img.shape(0), img.shape(1) - 1};
    for (int i=0; i<img.shape(0); ++i)
        for (int j=0; j<img.shape(1) - 1; ++j) {
            new_img(i, j) = img(i, j + 1) - img(i, j);
        }
    new_img.limit();
    return parseToObj(new_img);
}
