#include <cstring>
#include "fpng.h"

extern "C" {
    bool encode_to_file(const char* pFilename, const void* pImage, uint32_t w, uint32_t h, uint32_t num_chans, uint32_t flags = 0)
    {
        return fpng::fpng_encode_image_to_file(pFilename, pImage, w, h, num_chans, flags);
    }

    int get_info(const void* pImage, uint32_t image_size, uint32_t& width, uint32_t& height, uint32_t& channels_in_file)
    {
        return fpng::fpng_get_info(pImage, image_size, width, height, channels_in_file);
    }

    int decode_file(const char* pFilename, void* &vout, uint32_t& width, uint32_t& height, uint32_t& channels_in_file, uint32_t desired_channels) {
        // std::vector<uint8_t> out = reinterpret_cast<uint8_t*>(vout);
        std::vector<uint8_t> out;
        int ret = fpng::fpng_decode_file(pFilename, out, width, height, channels_in_file, desired_channels);
        if (ret == 0) {
            char* res = new char[out.size()];
            std::memcpy(res, out.data(), out.size()*sizeof(uint8_t));
            vout = res;
        }
        return ret;
    }

    int decode_memory(const void* pImage, uint32_t image_size, void* &vout, uint32_t& width, uint32_t& height, uint32_t& channels_in_file, uint32_t desired_channels) {
        std::vector<uint8_t> out;
        int ret = fpng::fpng_decode_memory(pImage, image_size, out, width, height, channels_in_file, desired_channels);
        if (ret == 0) {
            char* res = new char[out.size()];
            std::memcpy(res, out.data(), out.size()*sizeof(uint8_t));
            vout = res;
        }
        return ret;
    }

    void init() {
        fpng::fpng_init();
    }
}
