#ifndef TYPEDEFS_CUH
#define TYPEDEFS_CUH

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

    int16_t leftOffset;
    int16_t rightOffset;

    int16_t parentOffset;

} RBTreeBlockHeader;

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

#endif // TYPEDEFS_CUH