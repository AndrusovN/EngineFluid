#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>
#include <thread>

#include "Drawing.h"
#include "GameSetup.cuh"
    
const int WIDTH = 1024;
const int HEIGHT = 700;

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    UIDrawer* drawer = new UIDrawer(WIDTH, HEIGHT, "Nice drawer!", hInstance);

    std::thread mainThread(&startGame, drawer);
    mainThread.detach();

    drawer->processWindowEventsLoop();

    return 0;
}
