kernel void deviceAddressTranslationKernel(
    global void* p0, global void* p1, global void* p2, global void* p3,
    global void* p4, global void* p5, global void* p6, global void* p7,
    global void* p8, global void* p9, global void* p10, global void* p11,
    global void* p12, global void* p13, global void* p14, global void* p15,
    global uintptr_t dstBuffer[16])
{
    dstBuffer[0] = (uintptr_t)p0;
    dstBuffer[1] = (uintptr_t)p1;
    dstBuffer[2] = (uintptr_t)p2;
    dstBuffer[3] = (uintptr_t)p3;
    dstBuffer[4] = (uintptr_t)p4;
    dstBuffer[5] = (uintptr_t)p5;
    dstBuffer[6] = (uintptr_t)p6;
    dstBuffer[7] = (uintptr_t)p7;
    dstBuffer[8] = (uintptr_t)p8;
    dstBuffer[9] = (uintptr_t)p9;
    dstBuffer[10] = (uintptr_t)p10;
    dstBuffer[11] = (uintptr_t)p11;
    dstBuffer[12] = (uintptr_t)p12;
    dstBuffer[13] = (uintptr_t)p13;
    dstBuffer[14] = (uintptr_t)p14;
    dstBuffer[15] = (uintptr_t)p15;
}

kernel void sumKernel(global uintptr_t argArray[])
{
    global int* pOut = (global int*)argArray[0];
    global const int* pIn1 = (global int*)argArray[1];
    global const int* pIn2 = (global int*)argArray[2];

    const uint itemID = get_global_id(0);
    pOut[itemID] = pIn1[itemID] + pIn2[itemID];
}

