// OpenCLTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>

#define MAX_DEVICE_COUNT        8U
#define MAX_BUFFER_ARG_COUNT    16U

#ifdef _WIN32
#define FILE_BINARY_MODE    "b"
#else
#define FILE_BINARY_MODE
#endif // _WIN32


static void OpenCLTest(int deviceIndex)
{
    cl_uint numPlatforms = 0;               // the NO. of platforms
    cl_platform_id platform = NULL;         // the chosen platform
    cl_context context = NULL;              // OpenCL context
    cl_command_queue commandQueue = NULL;   // OpenCL command queue
    cl_program program = NULL;              // OpenCL kernel program object that'll be running on the compute device
    cl_mem input1MemObj = NULL;             // input1 memory object for input argument 1
    cl_mem input2MemObj = NULL;             // input2 memory object for input argument 2
    cl_mem outputMemObj = NULL;             // output memory object for output
    cl_mem dstDeviceMemPtrObj = NULL;       // Used to store the device-end memory addresses
    cl_kernel deviceAddrkernel = NULL;      // device address translation kernel object
    cl_kernel sumKernel = NULL;             // sum kernel object

    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS || numPlatforms == 0)
    {
        printf("clGetPlatformIDs failed: %d\nPlease check whether you have installed any OpenCL implementation\n", status);
        return;
    }
    // Just use the first platform
    status = clGetPlatformIDs(1, &platform, NULL);
    if (status != CL_SUCCESS)
    {
        puts("platform got failed and this should not happen!");
        return;
    }

    cl_uint numDevices = 0;
    cl_device_id devices[MAX_DEVICE_COUNT] = { NULL };
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (status != CL_SUCCESS || numDevices == 0)
    {
        printf("clGetDeviceIDs failed: %d\nNo GPU OpenCL devices found!!\n", status);
        return;
    }

    if (numDevices > MAX_DEVICE_COUNT)
        numDevices = MAX_DEVICE_COUNT;

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    if (status != CL_SUCCESS)
    {
        printf("clGetDeviceIDs failed: %d\nNo GPU device is found!\n", status);
        return;
    }

    // List all the devices
    char strBuf[256] = { '\0' };
    for (cl_uint i = 0; i < numDevices; i++)
    {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(strBuf), strBuf, NULL);
        printf("======== Device %d: %s ========\n", i, strBuf);

        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(strBuf), strBuf, NULL);
        printf("Driver version: %s\n", strBuf);

        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(strBuf), strBuf, NULL);
        printf("OpenCL version: %s\n", strBuf);
    }
    printf("\nCurrently selected device %d\n", deviceIndex);

    do
    {
        context = clCreateContext(NULL, 1, devices, NULL, NULL, &status);
        if (context == NULL || status != CL_SUCCESS)
        {
            printf("clCreateContext failed: %d\n", status);
            break;
        }

        commandQueue = clCreateCommandQueue(context, devices[deviceIndex], 0, &status);
        if (commandQueue == NULL || status != CL_SUCCESS)
        {
            printf("clCreateCommandQueue failed: %d\n", status);
            break;
        }

        FILE* fp = fopen("kernel.cl", "r" FILE_BINARY_MODE);
        if (fp == NULL)
        {
            puts("Cannot open file kernel.cl!");
            break;
        }
        fseek(fp, 0, SEEK_END);
        size_t fileLen = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        char* fileContent = malloc(fileLen + 1);
        if (fileContent == NULL || fileLen == 0)
        {
            fclose(fp);
            puts("Lack of memory to read kernel.cl...");
            break;
        }
        fread(fileContent, 1, fileLen, fp);
        fclose(fp);
        fileContent[fileLen] = '\0';

        program = clCreateProgramWithSource(context, 1, (const char* []){ fileContent },
            &fileLen, &status);
        if (program == NULL || status != CL_SUCCESS)
        {
            free(fileContent);
            printf("clCreateProgramWithSource failed: %d\n", status);
            break;
        }

        status = clBuildProgram(program, 1, &devices[deviceIndex], NULL, NULL, NULL);
        bool willQuit = status != CL_SUCCESS;
        while (willQuit)
        {
            printf("OpenCL kernel build error: %d\n", status);
            char *logBuf = NULL;
            size_t logLen = 0;
            status = clGetProgramBuildInfo(program, devices[deviceIndex], CL_PROGRAM_BUILD_LOG,
                logLen, logBuf, &logLen);
            if (status != CL_SUCCESS || logLen == 0)
                break;

            logBuf = malloc(logLen + 1);
            if (logBuf == NULL)
                break;

            clGetProgramBuildInfo(program, devices[deviceIndex], CL_PROGRAM_BUILD_LOG, logLen, logBuf, NULL);
            logBuf[logLen] = '\0';
            printf("%s\n", logBuf);

            free(logBuf);

            break;
        }
        free(fileContent);
        if (willQuit)
            break;

        sumKernel = clCreateKernel(program, "sumKernel", &status);
        if (sumKernel == NULL || status != CL_SUCCESS)
        {
            printf("clCreateKernel for sumKernel failed: %d\n", status);
            break;
        }

        size_t workgroupSize = 0;
        status = clGetKernelWorkGroupInfo(sumKernel, devices[deviceIndex], CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(workgroupSize), &workgroupSize, NULL);
        if (status != CL_SUCCESS)
        {
            printf("Query sumKernel workgroup size failed: %d\n", status);
            break;
        }

        const int nElems = 4096;
        const size_t bufferSize = nElems * sizeof(int);

        input1MemObj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, bufferSize,
            NULL, &status);
        if (input1MemObj == NULL || status != CL_SUCCESS)
        {
            printf("input1MemObj failed to create: %d\n", status);
            break;
        }
        input2MemObj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, bufferSize,
            NULL, &status);
        if (input2MemObj == NULL || status != CL_SUCCESS)
        {
            printf("input2MemObj failed to create: %d\n", status);
            break;
        }
        outputMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, bufferSize,
            NULL, &status);
        if (outputMemObj == NULL || status != CL_SUCCESS)
        {
            printf("outputMemObj failed to create: %d\n", status);
            break;
        }
        dstDeviceMemPtrObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
            MAX_BUFFER_ARG_COUNT * sizeof(uintptr_t), NULL, &status);
        if (dstDeviceMemPtrObj == NULL || status != CL_SUCCESS)
        {
            printf("dstDeviceMemPtrObj failed to create: %d\n", status);
            break;
        }

        cl_mem memPtrs[MAX_BUFFER_ARG_COUNT] = {
            outputMemObj, input1MemObj, input2MemObj, NULL
        };
        deviceAddrkernel = clCreateKernel(program, "deviceAddressTranslationKernel", &status);
        if (deviceAddrkernel == NULL || status != CL_SUCCESS)
        {
            printf("deviceAddrkernel failed to create: %d\n", status);
            break;
        }
        // Fetch the device-end addresses
        willQuit = false;
        for (unsigned i = 0; i < MAX_BUFFER_ARG_COUNT; i++)
        {
            status = clSetKernelArg(deviceAddrkernel, i, sizeof(cl_mem), (void*)&memPtrs[i]);
            if (status != CL_SUCCESS)
            {
                printf("deviceAddrkernel clSetKernelArg %u failed: %d\n", i, status);
                willQuit = true;
                break;
            }
        }
        if (willQuit)
            break;

        status = clSetKernelArg(deviceAddrkernel, MAX_BUFFER_ARG_COUNT, sizeof(cl_mem), 
            (void*)&dstDeviceMemPtrObj);
        if (status != CL_SUCCESS)
        {
            printf("deviceAddrkernel clSetKernelArg the last failed: %d\n", status);
            break;
        }

        status = clEnqueueNDRangeKernel(commandQueue, deviceAddrkernel, 1, NULL, (size_t[]) { 1 }, 
            (size_t[]) { 1 }, 0, NULL, NULL);
        if (status != CL_SUCCESS)
        {
            printf("Run kernel failed: %d\n", status);
            break;
        }

        // Do sum computing
        int* hostBuffer = malloc(bufferSize);
        if (hostBuffer == NULL)
        {
            puts("Lack of memory for hostBuffer!");
            break;
        }
        for (int i = 0; i < nElems; i++)
            hostBuffer[i] = i + 1;

        clEnqueueWriteBuffer(commandQueue, input1MemObj, CL_TRUE, 0, bufferSize, hostBuffer, 0, NULL, NULL);
        clEnqueueWriteBuffer(commandQueue, input2MemObj, CL_TRUE, 0, bufferSize, hostBuffer, 0, NULL, NULL);

        clSetKernelArg(sumKernel, 0, sizeof(cl_mem), (void*)&dstDeviceMemPtrObj);

        clEnqueueNDRangeKernel(commandQueue, sumKernel, 1, NULL, (size_t[]) { nElems },
            (size_t[]) { workgroupSize }, 0, NULL, NULL);

        puts("Verify the result...");
        clEnqueueReadBuffer(commandQueue, outputMemObj, CL_TRUE, 0, bufferSize, hostBuffer, 0, NULL, NULL);
        willQuit = false;
        for (int i = 0; i < nElems; i++)
        {
            if (hostBuffer[i] != (i + 1) * 2)
            {
                willQuit = true;
                printf("Error occurred @ %d\n", i);
                break;
            }
        }
        if (!willQuit)
            puts("Verification passed!");

    } while (false);

    if (sumKernel != NULL)
        clReleaseKernel(sumKernel);
    if (deviceAddrkernel != NULL)
        clReleaseKernel(deviceAddrkernel);

    if (input1MemObj != NULL)
        clReleaseMemObject(input1MemObj);
    if (input2MemObj != NULL)
        clReleaseMemObject(input2MemObj);
    if (outputMemObj != NULL)
        clReleaseMemObject(outputMemObj);
    if (dstDeviceMemPtrObj != NULL)
        clReleaseMemObject(dstDeviceMemPtrObj);

    if (program != NULL)
        clReleaseProgram(program);

    if (commandQueue != NULL)
        clReleaseCommandQueue(commandQueue);

    if (context != NULL)
        clReleaseContext(context);
}

int main(int argc, const char* argv[])
{
    uint32_t deviceIndex = 0;
    if (argc > 1)
    {
        deviceIndex = atoi(argv[1]);
        if (deviceIndex >= MAX_DEVICE_COUNT)
            deviceIndex = MAX_DEVICE_COUNT - 1;
    }

    OpenCLTest(deviceIndex);
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
