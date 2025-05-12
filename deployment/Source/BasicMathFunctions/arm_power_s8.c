#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"


/**
  @brief         Sum of the squares of the elements of an int8_t vector.
  @param[in]     pSrc       points to the input vector
  @param[in]     blockSize  number of samples in input vector
  @param[out]    pResult    sum of the squares value returned here
  
  @details       This function computes the sum of the squares of an int8_t vector
                 using a 32-bit accumulator to avoid overflow issues.
*/
void arm_power_s8(
  const int8_t * pSrc,
        uint32_t blockSize,
        int32_t input_offset,
        int32_t * pResult)
{
    uint32_t blkCnt;    /* Loop counter */
    int32_t sum = 0;    /* Accumulator */
    int32_t in;         /* Temporary input variable with offset applied */

    /* Loop unrolling: Process four elements at a time */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        in = (int32_t)(*pSrc++) + input_offset;
        sum += (in * in);

        in = (int32_t)(*pSrc++) + input_offset;
        sum += (in * in);

        in = (int32_t)(*pSrc++) + input_offset;
        sum += (in * in);

        in = (int32_t)(*pSrc++) + input_offset;
        sum += (in * in);

        blkCnt--;
    }

    /* Process remaining elements */
    blkCnt = blockSize % 4U;

    while (blkCnt > 0U)
    {
        in = (int32_t)(*pSrc++) + input_offset;
        sum += (in * in);

        blkCnt--;
    }

    /* Store result */
    *pResult = sum;
}