
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Drawing.h"

#include <stdio.h>
#include <Windows.h>
#include <thread>

const int WIDTH = 1024;
const int HEIGHT = 700;

cudaError_t renderWithCuda(Color *map, int time);
int main(UIDrawer* drawer);

__global__ void render(Color* map, int time)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    int red = 140 + (int)(110.0f * sinf((float)(x + time / 10) / 50.0f));
    int green = 140 + (int)(110.0f * sinf((float)(y + time / 5) / 100.0f));
    map[x + y * blockDim.x] = RGB(red, green, 30);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    UIDrawer* drawer = new UIDrawer(WIDTH, HEIGHT, "Nice drawer!", hInstance);

    std::thread mainThread(&main, drawer);
    mainThread.detach();

    drawer->processWindowEventsLoop();

    return 0;
}

int main(UIDrawer* drawer)
{
    int time = 0;
    while (true) {
        Color* map = new Color[WIDTH * HEIGHT];

        cudaError_t cudaStatus = renderWithCuda(map, time);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        drawer->resetColorMap(map);
        drawer->display();

        time++;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t renderWithCuda(Color* map, int time)
{
    Color* device_map = NULL;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&device_map, HEIGHT * WIDTH * sizeof(Color));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    render<<<HEIGHT, WIDTH>>>(device_map, time);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(map, device_map, HEIGHT * WIDTH * sizeof(Color), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(device_map);
    
    return cudaStatus;
}
