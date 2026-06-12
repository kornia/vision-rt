/* CUDA texture object helpers for the trt-preproc letterbox kernel.
   Creates a pitch-2D cudaTextureObject_t over RGBA u8 device memory so the
   nvrtc kernel can use hardware bilinear sampling instead of a software BILERP. */

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <cstring>

extern "C" {

/* Create a pitch-2D texture over an RGBA u8 device buffer.
   readMode  = cudaReadModeNormalizedFloat → samples return float4 in [0,1].
   filterMode = cudaFilterModeLinear       → hardware bilinear.
   addressMode = cudaAddressModeBorder     → out-of-bounds returns {0,0,0,0}
                 (kernel handles gray fill manually for YOLO letterbox).
   Returns cudaError_t (0 = success). */
int preproc_create_tex2d(
    void*    dev_ptr,
    uint32_t width,
    uint32_t height,
    uint32_t pitch,           /* bytes per row, including alignment padding */
    uint64_t* tex_obj_out
) {
    if (!dev_ptr || width == 0 || height == 0 || pitch == 0 || !tex_obj_out)
        return 1;

    cudaResourceDesc rdesc;
    memset(&rdesc, 0, sizeof(rdesc));
    rdesc.resType                   = cudaResourceTypePitch2D;
    rdesc.res.pitch2D.devPtr        = dev_ptr;
    rdesc.res.pitch2D.desc          = cudaCreateChannelDesc(8, 8, 8, 8,
                                          cudaChannelFormatKindUnsigned);
    rdesc.res.pitch2D.width         = width;
    rdesc.res.pitch2D.height        = height;
    rdesc.res.pitch2D.pitchInBytes  = pitch;

    cudaTextureDesc tdesc;
    memset(&tdesc, 0, sizeof(tdesc));
    tdesc.addressMode[0]  = cudaAddressModeBorder;
    tdesc.addressMode[1]  = cudaAddressModeBorder;
    tdesc.filterMode      = cudaFilterModeLinear;
    tdesc.readMode        = cudaReadModeNormalizedFloat;
    tdesc.normalizedCoords = 0;  /* unnormalized (pixel) coordinates */

    cudaTextureObject_t tex = 0;
    cudaError_t err = cudaCreateTextureObject(&tex, &rdesc, &tdesc, nullptr);
    if (err != cudaSuccess) return (int)err;

    *tex_obj_out = (uint64_t)tex;
    return 0;
}

/* Destroy a texture object created by preproc_create_tex2d.
   Call only after the consuming kernel has finished (stream synced). */
void preproc_destroy_tex2d(uint64_t tex_obj) {
    if (tex_obj) cudaDestroyTextureObject((cudaTextureObject_t)tex_obj);
}

} // extern "C"
