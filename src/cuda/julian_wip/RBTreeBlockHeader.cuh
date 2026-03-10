#pragma once
#include <stdint.h>

enum Color : uint8_t { RED, BLACK };

class RBTreeBlockHeader {
    /*
    first bit will be full status, second bit will be color
    typedef struct{
        int16_t fullStatus : 1
        int16_t color : 1
        int16_t size : 14
    } fullSize
    */

    private:
        int16_t fullSize;
        int16_t leftOffset;
        int16_t rightOffset;
        int16_t parentOffset;

    public:

        __device__ RBTreeBlockHeader(int16_t fullSize,int16_t leftOffset, int16_t rightOffset, int16_t parentOffset)
            :fullSize(fullSize), leftOffset(leftOffset),rightOffset(rightOffset),parentOffset(parentOffset){}

        __device__ int16_t size() {
            return fullSize & 0x3FFF;
        }

        __device__ bool isFull() {
            return fullSize < 0;
        }

        __device__ Color color() {
            return (Color)((fullSize >> 14) & 0x1);
        }

        __device__ void setColor(Color c) {
            if (c == RED) {
                fullSize &= ~0x4000; // Clear bit 14
            } else {
                fullSize |= 0x4000;  // Set bit 14
            }
        }

        __device__ void setFull(bool full) {
            if (full) {
                fullSize |= 0x8000;  // Set bit 15
            } else {
                fullSize &= ~0x8000; // Clear bit 15
            }
        }

        __device__ void setSize(int16_t s) {
            // Preserve the upper 2 bits, set lower 14 bits
            fullSize = (fullSize & 0xC000) | (s & 0x3FFF);
        }

        __device__ int16_t calculateOffset(unsigned char* to) {
            return to - (unsigned char*)this;
        }

        __device__ void set_right_offset(RBTreeBlockHeader* input){
            rightOffset = input ? calculateOffset((unsigned char*)input) : 0;
        }

        __device__ void set_left_offset(RBTreeBlockHeader* input){
            leftOffset = input ? calculateOffset((unsigned char*)input) : 0;
        }

        __device__ void set_parent_offset(RBTreeBlockHeader* input){
            parentOffset = input ? calculateOffset((unsigned char*)input) : 0;
        }

        __device__ BlockFooter* footer() {
            return (RBTreeBlockHeader*)((unsigned char*)this + size() + sizeof(RBTreeBlockHeader));
        }

        __device__ BlockFooter* prevFooter() {
            return (BlockFooter*)((unsigned char*)this - sizeof(BlockFooter));
        }

        __device__ RBTreeBlockHeader* debugPrevHeader() {
            uint threadIndex = get_linear_thread_index();

            if((void*)this == (void*)memPools[threadIndex].base() ){
                return NULL;
            }

            RBTreeBlockHeader* prevHeader = prevFooter()->RBTreeheader();

            return prevHeader;
        }

        __device__ RBTreeBlockHeader* debugNextHeader(){
            uint threadIndex = get_linear_thread_index();
            uint threadPoolSize = get_thread_pool_size();

            int currentHeaderOffset = calculateOffset(memPools[threadIndex].base());

            if((currentHeaderOffset + size() + sizeof(RBTreeBlockHeader) + sizeof(BlockFooter)) >= threadPoolSize){
                return NULL;
            }

            return (RBTreeBlockHeader*)((unsigned char*)this + size() + sizeof(BlockFooter) + sizeof(RBTreeBlockHeader));
        }

        __device__ RBTreeBlockHeader* rightHeader(){
            if(rightOffset == 0){
                return NULL;
            }
            return (RBTreeBlockHeader*)((unsigned char*)this + rightOffset);
        }

        __device__ RBTreeBlockHeader* parentHeader(){
            if(parentOffset == 0){
                return NULL;
            }
            return (RBTreeBlockHeader*)((unsigned char*)this + parentOffset);
        }

        __device__ RBTreeBlockHeader* leftHeader(){
            if(leftOffset == 0){
                return NULL;
            }
            return (RBTreeBlockHeader*)((unsigned char*)this + leftOffset);
        }
};
