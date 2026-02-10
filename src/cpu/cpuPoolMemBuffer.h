#include <stdint.h>
#include <stddef.h>

#ifndef BLOCK_H
#define BLOCK_H


/*

BlockHeader for a RBTree impl (for later)

typedef struct {

    uint64_t full           : 1;
    uint64_t color          : 1;

    // this can be any number of bits >= 12,
    // iff the size of each threads buffer is < 4KB
    uint64_t size           : 14; 

    int64_t leftOffset      : 16;
    int64_t rightOffset     : 16;

    int64_t parentOffset    : 16;

} BlockHeader;
*/

// typedef struct {

//     uint64_t full           : 1;
//     uint64_t size           : 15;

//     int64_t nextOffset      : 16;
//     int64_t prevOffset      : 16;

//     int64_t padding         : 16;

// } BlockHeader;

typedef struct {

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

#endif