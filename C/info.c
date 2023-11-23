#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const size_t nr_plat = 10;
const size_t nr_dev = 10;

const char *device_name(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_DEFAULT:
            return "DEFAULT";
        case CL_DEVICE_TYPE_CPU:
            return "CPU";
        case CL_DEVICE_TYPE_GPU:
            return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR:
            return "ACCELERATOR";
        case CL_DEVICE_TYPE_ALL:
            return "ALL";
        default:
            return "INVALID-DEFAULT";
    }
}

void display_device(char *name, cl_bool available, cl_device_type type) {
    printf("device name: %s\n", name);
    printf("is available ? %s\n", (available == CL_TRUE ? "yes" : "no"));
    printf("type: %s\n", device_name(type));
}

int main(void) {
    cl_platform_id *platforms = (cl_platform_id *)calloc(nr_plat, sizeof(cl_platform_id));
    if (platforms == NULL) {
        fprintf(stderr, "Can't allocate memory for %zu cl_platform_id\n", nr_plat);
        return 1;
    }
    cl_uint actual_nr_plat = 0;

    cl_int ret = 0;

    ret = clGetPlatformIDs(nr_plat, platforms, &actual_nr_plat);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Error while getting platform ids: %d", ret);
        free(platforms);
        return 1;
    }

    printf("Got %u platform ids:\n", actual_nr_plat);
    for (cl_uint i = 0; i < actual_nr_plat; i++) {
        cl_platform_id platform = platforms[i];

        cl_device_id *devices = (cl_device_id *)calloc(nr_dev, sizeof(cl_device_id));
        if (devices == NULL) {
            fprintf(stderr, "Can't allocate memory for %zu devices for platform %u\n", nr_dev, i);
            free(platforms);
            return 1;
        }

        cl_uint actual_nr_dev = 0;
        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, nr_dev, devices, &actual_nr_dev);
        if (ret != CL_SUCCESS) {
            fprintf(stderr, "Error while getting devices for platform %u: %d\n", i, ret);
            free(devices);
            free(platforms);
            return 1;
        }

        printf("Platform %u has %u devices\n", i, actual_nr_dev);

        for (cl_uint j = 0; j < actual_nr_dev; j++) {
            cl_device_id device = devices[j];

            cl_device_type type;
            ret = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
            if (ret != CL_SUCCESS) {
                fprintf(stderr, "Can't query device %u type on platform %u: %d\n", j, i, ret);
                free(devices);
                free(platforms);
                return 1;
            }

            cl_bool available;
            ret = clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL);
            if (ret != CL_SUCCESS) {
                fprintf(stderr, "Can't query device %u availability on platform %u: %d\n", j, i, ret);
                free(devices);
                free(platforms);
                return 1;
            }

            char *name = (char *)calloc(100, sizeof(char));
            if (name == NULL) {
                fprintf(stderr, "Can't allocate memory for name of device %u on platform %u\n", j, i);
                free(devices);
                free(platforms);
                return 1;
            }

            size_t len = 0;
            ret = clGetDeviceInfo(device, CL_DEVICE_NAME, 100 * sizeof(char), name, &len);
            if (ret != CL_SUCCESS) {
                fprintf(stderr, "Can't query device %u name on platform %u: %d\n", j, i, ret);
                free(name);
                free(devices);
                free(platforms);
                return 1;
            }

            display_device(name, available, type);
            free(name);
        }

        free(devices);

    }

    free(platforms);

    return 0;
}
