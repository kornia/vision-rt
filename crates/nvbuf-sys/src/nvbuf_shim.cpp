/* NvBufSurface NVMM shim — geometry helpers + CUDA external memory import.
   On Jetson Orin, NVBUF_MEM_DEFAULT = NVBUF_MEM_SURFACE_ARRAY.
   surfaceList[0].dataPtr is NOT valid for SURFACE_ARRAY; use the DMA-BUF FD
   from surfaceList[0].bufferDesc and import it via cudaImportExternalMemory. */

#include <nvbufsurface.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

extern "C" {

/* DEPRECATED — dataPtr is documented "not valid for NVBUF_MEM_SURFACE_ARRAY".
   Use nvbuf_cuda_import() instead.  Kept only for NVBUF_MEM_CUDA_UNIFIED code. */
void* nvbuf_cuda_ptr(const void* surf) {
    if (!surf) return nullptr;
    const NvBufSurface* s = static_cast<const NvBufSurface*>(surf);
    if (s->numFilled == 0) return nullptr;
    return s->surfaceList[0].dataPtr;
}

uint32_t nvbuf_width(const void* surf) {
    if (!surf) return 0;
    const NvBufSurface* s = static_cast<const NvBufSurface*>(surf);
    return s->numFilled > 0 ? s->surfaceList[0].width : 0;
}

uint32_t nvbuf_height(const void* surf) {
    if (!surf) return 0;
    const NvBufSurface* s = static_cast<const NvBufSurface*>(surf);
    return s->numFilled > 0 ? s->surfaceList[0].height : 0;
}

uint32_t nvbuf_pitch(const void* surf) {
    if (!surf) return 0;
    const NvBufSurface* s = static_cast<const NvBufSurface*>(surf);
    return s->numFilled > 0 ? s->surfaceList[0].pitch : 0;
}

/* Returns the DMA-BUF file descriptor for the first surface (-1 on error).
   Valid only for NVBUF_MEM_SURFACE_ARRAY / NVBUF_MEM_HANDLE. */
int nvbuf_dmabuf_fd(const void* surf) {
    if (!surf) return -1;
    const NvBufSurface* s = static_cast<const NvBufSurface*>(surf);
    if (s->numFilled == 0) return -1;
    uint64_t desc = s->surfaceList[0].bufferDesc;
    if (desc > (uint64_t)INT_MAX) return -1;  /* L2: guard truncation */
    return (int)desc;
}

/* Returns the memory layout: 0 = NVBUF_LAYOUT_PITCH, 1 = NVBUF_LAYOUT_BLOCK_LINEAR.
   Only PITCH layout is valid for CUDA linear device pointer access. */
int nvbuf_layout(const void* surf) {
    if (!surf) return -1;
    const NvBufSurface* s = static_cast<const NvBufSurface*>(surf);
    if (s->numFilled == 0) return -1;
    return (int)s->surfaceList[0].layout;
}

/* Total allocated bytes for the first surface.
   Uses dataSize when available; falls back to pitch×height (pitch-linear only). */
uint64_t nvbuf_data_size(const void* surf) {
    if (!surf) return 0;
    const NvBufSurface* s = static_cast<const NvBufSurface*>(surf);
    if (s->numFilled == 0) return 0;
    uint32_t ds = s->surfaceList[0].dataSize;
    if (ds > 0) return (uint64_t)ds;
    return (uint64_t)s->surfaceList[0].pitch * s->surfaceList[0].height;
}

/* Import a DMA-BUF FD as a CUDA external memory object, then map a device ptr.
   Returns cudaError_t (0 = success).
   Internally dup()s `fd` — caller retains ownership of the original FD.
   On success, *ext_mem_out and *dev_ptr_out are valid CUDA handles to be
   released via nvbuf_cuda_release() after the consuming stream is synced. */
int nvbuf_cuda_import(int fd, uint64_t size,
                      void** ext_mem_out, void** dev_ptr_out) {
    if (fd < 0 || size == 0 || !ext_mem_out || !dev_ptr_out) return 1;

    int dup_fd = dup(fd);
    if (dup_fd < 0) return 2;

    cudaExternalMemoryHandleDesc hdesc;
    memset(&hdesc, 0, sizeof(hdesc));
    hdesc.type       = cudaExternalMemoryHandleTypeOpaqueFd;
    hdesc.handle.fd  = dup_fd;
    hdesc.size       = size;
    hdesc.flags      = 0;

    cudaExternalMemory_t ext_mem = nullptr;
    cudaError_t err = cudaImportExternalMemory(&ext_mem, &hdesc);
    if (err != cudaSuccess) {
        close(dup_fd); /* CUDA did not consume it on failure */
        return (int)err;
    }

    cudaExternalMemoryBufferDesc bdesc;
    memset(&bdesc, 0, sizeof(bdesc));
    bdesc.offset = 0;
    bdesc.size   = size;
    bdesc.flags  = 0;

    void* dev_ptr = nullptr;
    err = cudaExternalMemoryGetMappedBuffer(&dev_ptr, ext_mem, &bdesc);
    if (err != cudaSuccess) {
        cudaDestroyExternalMemory(ext_mem);
        return (int)err;
    }

    *ext_mem_out = (void*)ext_mem;
    *dev_ptr_out = dev_ptr;
    return 0;
}

/* Release handles from nvbuf_cuda_import.  Call only after stream sync. */
void nvbuf_cuda_release(void* ext_mem, void* dev_ptr) {
    if (dev_ptr) cudaFree(dev_ptr);
    if (ext_mem) cudaDestroyExternalMemory((cudaExternalMemory_t)ext_mem);
}

} // extern "C"
