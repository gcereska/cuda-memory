typedef struct {

    
    /*
    first bit will be full status, second bit will be color 
    typedef struct{
        int16_t fullStatus : 1
        int16_t color : 1
        int16_t size : 14
    } fullSize

    */
    int16_t fullSize;

    int16_t nextOffset;
    int16_t prevOffset;

    int16_t padding;

} BlockHeader;

typedef struct {

    /*while 6 byte accesses are bad, 2 bytes should be ok because they will still be fully accessed in one word. 
    I will prob pad the size to align the total allocation to 4 bytes however*/
    
    int16_t headerOffset;
} BlockFooter;

typedef struct {
    unsigned char* memBuffer;
    BlockHeader *freeList;
    BlockHeader *fullList;

    uint threadPoolSize;
} MemBufferStorage;



__device__ void init_gpu_buffer(uint sharedMemSize);

__device__ void *cmalloc(unsigned long size);

__device__ void cfree(void *ptr);

__device__ void debug_print_buffer();
__device__ void debug_print_free_list();
__device__ void debug_print_full_list();

// __device__ void printlayout();

// __device__ void printbytes();

// __device__ int dataBytes(BlockHeader *head);

// __device__ int headerBytes(BlockHeader *head);