import ctypes
import os
from typing import Optional, Tuple 
import numpy as np
import cv2
import timeit
import numpy.typing as npt
import glob

# Load extension
dir_path = os.path.dirname(os.path.realpath(__file__))
so_files = glob.glob('libpyfpng.*.so')
if len(so_files) > 1:
    raise RuntimeError(f"Found multiple shared object libraries: {so_files}")
elif not so_files:
    raise RuntimeError(f"Failed to locate libpyfpng library!")
else:
    so_lib = so_files[0]
    handle = ctypes.CDLL(os.path.join(dir_path, so_lib))

handle.encode_to_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
handle.encode_to_file.restype = ctypes.c_bool
handle.init()

#handle.decode_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
#handle.decode_file.restype = ctypes.c_int

def init():
    handle.init()
  
def encode_to_file(file_path: str, data: npt.NDArray[np.uint8]):
    h, w, c = data.shape
    flags = 0
    filename = file_path.encode('utf-8')

    rgb_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    pointer = rgb_data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

    return handle.encode_to_file(filename, pointer, w, h, c, flags)    


def decode_file_to_memory(file_path: str, desired_channels:int=3) -> Tuple[int, Optional[npt.NDArray[np.uint8]]]:
    filename = file_path.encode('utf-8')
    h, w, c = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
    out = ctypes.c_void_p()

    ret = handle.decode_file(filename, ctypes.byref(out), ctypes.byref(w), ctypes.byref(h), ctypes.byref(c), desired_channels)
    if ret == 0:
        Darr = ctypes.c_uint8 * w.value * h.value * c.value * ctypes.sizeof(ctypes.c_uint8)
        arr_data = Darr.from_address(out.value)
        res = np.ctypeslib.as_array(arr_data)[0].reshape(h.value, w.value, c.value).copy()
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        handle.free(out)
    else:
        res = None

    return ret, res

def decode_to_memory(data: bytes, desired_channels:int=3):
    h, w, c = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
    out = ctypes.c_void_p()

    size = len(data)
    char_array = ctypes.c_char * size
    pointer = char_array.from_buffer(bytearray(data))

    ret = handle.decode_memory(pointer, size, ctypes.byref(out), ctypes.byref(w), ctypes.byref(h), ctypes.byref(c), desired_channels)
    if ret == 0:
        Darr = ctypes.c_uint8 * w.value * h.value * c.value * ctypes.sizeof(ctypes.c_uint8)
        arr_data = Darr.from_address(out.value)
        res = np.ctypeslib.as_array(arr_data)[0].reshape(h.value, w.value, c.value).copy()
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        handle.free(out)
    else:
        res = None

    return ret, res

def get_file_info(data: bytes) -> Tuple[int, int, int, int]:
    h, w, c = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()

    size = len(data)
    char_array = ctypes.c_char * size
    pointer = char_array.from_buffer(bytearray(data))

    ret = handle.get_info(pointer, size, ctypes.byref(w), ctypes.byref(h), ctypes.byref(c))

    # ret:
    # 0  -  file is a valid PNG file and written by FPNG and the decode succeeded
    # 1  -  file is a valid PNG file, but it wasn't written by FPNG so you should try decoding it with a general purpose PNG decoder
    # 2> - invalid file - see fpng.h for details
    return ret, h.value, w.value, c.value

def read_image(file_path: str):
    with open(file_path, 'rb') as f:
        data = f.read()

    img = None
    ret, _, _, _ = get_file_info(data)
    if ret == 0:
        ret_read, img = decode_to_memory(data)
    
    if img is None:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img
