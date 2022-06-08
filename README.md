# Python Bindings for `fpng`

Implemets python interface to call fpng functions.

Note that the images written / read are in RGB channel order, in contrast to OpenCV
that reads / writes files in BGR format.

If you know how to avoid reserving memory twice (once in C++ side and once in Python side)
when decoding images (see src/fpng_py.cpp and pyfpng.decode_to_memory function), please
let me know or submit a pull request.


## Installation
1. Clone repository `git clone --recurse-submodules git@gitlab.com:labra-public/pyfpng.git`
2. Install `python setup.py install`

## Usage
```
import pyfpng      # Calls fpng::init() on import

# Encode numpy array and write to file:
data = np.zeros((512, 512, 3), dtype=np.uint8)
success: bool = pyfpng.encode_to_file('sample.png', data)

# Decode file to memory
success: int # see fpng.h
data: Optional[NDArray[np.uint8]]
success, data = pyfpng.decode_file_to_memory('sample.png')

# Decode bytes to memory
with open('sample.png', 'rb') as f:
    bytes_data = f.read()

success, data = pyfpng.decode_to_memory(bytes_data)

# Get info about file (e.g. if it can be decoded using pyfpng)
ret, height, widht, channels = pyfpng.get_file_info(bytes_data)
if ret == 0:
    # can decode...
    ...
else:
    # see fpng.h for details on error code
    ...

# Check if can decode file and decode it if possible
success, img = read_image('sample.png)
```

## Performance
In tests, beats cv2.imwrite / cv2.imencode by ~5x, with similar result file size. 
For cv2.imread/cv2.imdecode, ~10-20% faster.

## License
Licensed under [Unlicense](https://unlicense.org/)
