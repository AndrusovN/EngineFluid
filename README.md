# EngineFluid
Raytracing 3D engine to simulate fluid dynamics around obstacles

This is a 3D raytracing engine on NVIDIA CUDA and windows.h libraries, developed to simulate fluid flows around obstacles.
It is developed to simulate aircraft flight and air fluxes behaviour to improve aircraft's parameters.

Now it's in development stage, so currently it has only simple UI window.

## Files & Classes
  1. Drawer.cpp, Drawer.h - UIDrawer - basic UI functions. It has colorMap - array of colors a.k.a. screen :) and provides some functions with it.
  2. kernel.cu - main file, which starts the program
