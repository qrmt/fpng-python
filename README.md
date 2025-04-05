# Python Bindings for `fpng`

Implemets Python interface to call [fpng](https://github.com/richgel999/fpng) functions.
`fpng` is a super fast C++ .PNG writer/reader.

Note that the images written / read are in RGB channel order, in contrast to OpenCV
that reads / writes files in BGR format.


## Installation
1. Clone repository `git clone --recurse-submodules git@github.com:qrmt/fpng-python.git`
2. Install `python setup.py install`

## Usage
```
import pyfpng      # Calls fpng::init() on import
import numpy as np
from typing import Optional

# Encode numpy array and write to file:
data = np.zeros((512, 512, 3), dtype=np.uint8)
success: bool = pyfpng.encode_image_to_file('sample.png', data)
print(f"Encode to file success: {success}")

# Encode numpy array to memory
success: bool
encoded: Optional[bytes]
success, encoded = pyfpng.encode_image_to_memory(data)
print(f"Encode to memory success: {success}")

# Decode file to memory
success: int # see fpng.h
success, data = pyfpng.decode_file('sample.png')
print(f"Decode file code: {success}")

# Decode bytes to memory
with open('sample.png', 'rb') as f:
    bytes_data = f.read()

success, data = pyfpng.decode_memory(bytes_data)
print(f"Decode memory: {success}")

# Get info about file (e.g. if it can be decoded using pyfpng)
ret, height, width, channels = pyfpng.get_info(bytes_data)
print(f"Read info: {width}x{height},{channels}")

if ret == 0:
    # can decode...
    ...
else:
    # see fpng.h for details on error code
    ...

```

## Performance
In tests, beats cv2.imwrite / cv2.imencode by ~5x, with similar result file size. 
For cv2.imread/cv2.imdecode, ~10-20% faster.

## License
Licensed under [Unlicense](https://unlicense.org/)
