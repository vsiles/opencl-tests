#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define DIM (8)
#define LIST_SIZE (DIM * DIM)

#define SOURCE                                          \
"__kernel void entry_point(__global int *C) {\n"        \
"    // Get the indexes of the current work items\n"    \
"    int i = get_global_id(0);\n"                       \
"    int j = get_global_id(1);\n"                       \
"\n"                                                    \
"    // Store a 1 at the relevant location\n"           \
"    C[i + 8 * j] = (i << 8 | j);\n"                    \
"}\n"

#define SOURCE_LEN (sizeof(SOURCE))

static const char* source_str = SOURCE;

int main(void) {
    size_t source_size = SOURCE_LEN;

    // Get platform and device information
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
            &device_id, &ret_num_devices);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Can't get Device IDS (GPU): %d\n", ret);
        return 1;
    }


    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create context: %d\n", ret);
        return 1;
    }

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create a command queue: %d\n", ret);
        clReleaseContext(context);
        return 1;
    }

    // Create memory buffers on the device for each vector
    // TODO: learn about host_ptr ?
    cl_mem res_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            LIST_SIZE * sizeof(unsigned int), NULL, &ret);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create a buffer: %d\n", ret);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return 1;
    }

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program: %d\n", ret);
        clReleaseMemObject(res_obj);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return 1;
    }

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to build program: %d\n", ret);

    int status;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS,
                                sizeof(int),
                                &status, 
                                NULL);

    printf("BuildInfo: %d\n", ret);
    printf("status: %d\n", status);

    char buffer[1024] = { 0 };
    size_t actual;

    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                1024,
                                (char *)buffer, 
                                &actual);

    printf("BuildInfo: %d\n", ret);
    printf("log:\n%s\n", buffer);





        clReleaseProgram(program);
        clReleaseMemObject(res_obj);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return 1;
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "entry_point", &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel: %d\n", ret);
        clReleaseProgram(program);
        clReleaseMemObject(res_obj);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return 1;
    }

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&res_obj);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arguments: %d\n", ret);
        clReleaseProgram(program);
        clReleaseMemObject(res_obj);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return 1;
    }

    // Execute the OpenCL kernel on the list
    size_t global_work_size[2] = {8, 8};
    size_t local_work_size[2] = {2, 2} ; // Divide work items into groups of 2 
    ret = clEnqueueNDRangeKernel(command_queue,     // valid command-queue
                                 kernel,            // valid kernel.
                                                    // context associated with kernel and command-queue must be the same
                                 2,                 // 0 < work_dim <= 3
                                 NULL,              // global_work_offset, must be NULL for now
                                 (size_t *) global_work_size, // array of work_dim unsigned values that describe the
                                                              // number of global work-items in work_dim dimensions
                                 (size_t *) local_work_size,   // see doc, must divide global_work_size
                                 0,                 // size event wait list
                                 NULL,              // event wait list
                                 NULL               // returned event
                                );

    // Read the memory buffer C on the device to the local variable C
    int *C = (int *)malloc(sizeof(unsigned int) * LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, res_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(unsigned int), C, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to read back result from device: %d\n", ret);
        clReleaseProgram(program);
        clReleaseMemObject(res_obj);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return 1;
    }

    // Display the result to the screen
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            unsigned int c = C[i + DIM * j];
            unsigned int x = (c >> 8) & 0xff;
            unsigned int y = c & 0xff;
            /* printf("i= %d, j= %d, x= %u, y= %u, c = 0x%x\n", i, j, x, y, c); */
            printf("0x%x ", c);
        }
        printf("\n");
    }
    free(C);

    // Clean up
    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(res_obj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
