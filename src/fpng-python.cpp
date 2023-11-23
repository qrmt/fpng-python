#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "fpng.h"

static PyObject* fpng_init(PyObject *self, PyObject *args) {
    fpng::fpng_init();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* fpng_encode_image_to_memory(PyObject *self, PyObject *args)
{

    PyArrayObject *arrData;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrData)){
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. RGB uint8 numpy array as argument!");
        return NULL;
    }

    if (NULL == arrData){
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_NDIM(arrData) != 3){
        PyErr_SetString(PyExc_ValueError, "Array NDIM > 3");
        return NULL;
    }

    if (PyArray_TYPE(arrData) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "Invalid array type ");
        return NULL;
    }

    unsigned int flags = 0;
    std::vector<uint8_t> out;

    bool success = fpng::fpng_encode_image_to_memory(PyArray_DATA(arrData), PyArray_DIM(arrData, 1), PyArray_DIM(arrData, 0), PyArray_DIM(arrData, 2), out, flags);

    PyObject* encoded;
    if (success) {
        char* bytes = new char[out.size()];
        std::copy(out.begin(),out.end(),bytes);
        encoded = PyBytes_FromStringAndSize(bytes, out.size());

    } else {
        Py_INCREF(Py_None);
        encoded = Py_None;
    }
    return PyTuple_Pack(2, PyBool_FromLong(success), encoded);
}

static PyObject* fpng_encode_image_to_file(PyObject *self, PyObject *args)
{
    PyArrayObject *arrData;
    unsigned char *file_path;
    if (!PyArg_ParseTuple(args, "sO!", &file_path, &PyArray_Type, &arrData)){
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Pass file destination and RGB uint8 numpy array as arguments!");
        return NULL;
    }

    if (NULL == arrData){
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_NDIM(arrData) != 3){
        PyErr_SetString(PyExc_ValueError, "Array NDIM > 3");
        return NULL;
    }

    if (PyArray_TYPE(arrData) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "Invalid array type - should be np.uint8");
        return NULL;
    }

    unsigned int flags = 0;

    bool success = fpng::fpng_encode_image_to_file((const char*)file_path, PyArray_DATA(arrData), PyArray_DIM(arrData, 1), PyArray_DIM(arrData, 0), PyArray_DIM(arrData, 2), flags);
    return PyBool_FromLong(success);
}

static PyObject* fpng_get_info(PyObject *self, PyObject *args)
{
    const char* image_bytes;
    uint32_t image_size;

    if (!PyArg_ParseTuple(args, "y#", &image_bytes, &image_size)){
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Pass file bytes as argument!");
        return NULL;
    }
    
    uint32_t width, height, channels;
    int retcode = fpng::fpng_get_info(image_bytes, image_size, width, height, channels);

    return Py_BuildValue("(iiii)", retcode, (int)width, (int)height, (int)channels);    
}

static PyObject* fpng_decode_file(PyObject *self, PyObject *args)
{
    const char *file_path;
    if (!PyArg_ParseTuple(args, "s", &file_path)){
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Pass file path as argument!");
        return NULL;
    }

    std::vector<uint8_t> out;
    uint32_t width, height, channels;
    unsigned int desired_channels = 3;
    int ret = fpng::fpng_decode_file(file_path, out, width, height, channels, desired_channels);

    PyObject* img;
    if (ret == 0) {
        npy_intp dims[3] = {height, width, channels};
        PyArrayObject* numpyArray = (PyArrayObject*)PyArray_SimpleNewFromData(3, dims, NPY_UINT8, (uint8_t*)out.data());
        img = PyArray_Return(numpyArray);
    } else {
        Py_INCREF(Py_None);
        img = Py_None;
    }
    return PyTuple_Pack(2, PyLong_FromLong(ret), img);
}

static PyObject* fpng_decode_memory(PyObject *self, PyObject *args)
{
    const char* image_bytes;
    uint32_t image_size;

    if (!PyArg_ParseTuple(args, "y#", &image_bytes, &image_size)){
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Pass file bytes as argument!");
        return NULL;
    }
    
    std::vector<uint8_t> out;
    uint32_t width, height, channels;
    unsigned int desired_channels = 3;
    int ret = fpng::fpng_decode_memory(image_bytes, image_size, out, width, height, channels, desired_channels);

    PyObject* img;
    if (ret == 0) {
        npy_intp dims[3] = {height, width, channels};
        PyArrayObject* numpyArray = (PyArrayObject*)PyArray_SimpleNewFromData(3, dims, NPY_UINT8, (uint8_t*)out.data());
        img = PyArray_Return(numpyArray);
    } else {
        Py_INCREF(Py_None);
        img = Py_None;
    }

    return PyTuple_Pack(2, PyLong_FromLong(ret), img);
}


static PyMethodDef FpngMethods[] = {
    {"encode_image_to_file", fpng_encode_image_to_file, METH_VARARGS, "Encode numpy array to file as png using fpng."},
    {"encode_image_to_memory", fpng_encode_image_to_memory, METH_VARARGS, "Encode numpy array to memory as png using fpng."},
    {"decode_file", fpng_decode_file, METH_VARARGS, "Decode file from png to numpy array using fpng."},
    {"decode_memory", fpng_decode_memory, METH_VARARGS, "Decode png file from bytes."},
    {"get_info", fpng_get_info, METH_VARARGS, "Get png info about bytes. Returns fpng decode success/error indicator, along with width, height, channel data."},
    {"init", fpng_init, METH_VARARGS, "Initialize fpng."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef pyfpngmodule =
{
    PyModuleDef_HEAD_INIT,
    "pyfpng",
    "Python bindings for fpng",
    -1,
    FpngMethods
};

PyMODINIT_FUNC
PyInit_pyfpng(void)
{
    import_array();
    fpng::fpng_init();
    return PyModule_Create(&pyfpngmodule);
}
