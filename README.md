# EngineFluid
Raytracing 3D engine to simulate fluid dynamics around obstacles

This is a 3D raytracing engine on NVIDIA CUDA and windows.h libraries, developed to simulate fluid flows around obstacles.
It is developed to simulate aircraft flight and air fluxes behaviour to improve aircraft's parameters.

Now it's in development stage, so currently it has not all functionality.

## Files & Classes
  1. Drawer.cpp, Drawer.h - UIDrawer - basic UI functions. It has colorMap - array of colors a.k.a. screen :) and provides some functions with it
  2. kernel.cu - main file, which starts the program
  3. assert.cuh - CUDA-compatible assertions
  4. Camera.cu, Camera.cuh - Basic 3D Camera component - renders on GPU
  5. Component.cuh - base component class. Every component should be inherited from Component and realize methods typeId, moveToDevice and moveToHost
  6. GameManager.cu, GameManager.cuh - Manager of the game. Switches between scenes, runs game loop, calls update and other component methods, GameManager is singleton
  7. GameObject.cu, GameObject.cuh - GameObject class. It's meant that there are no inherited classes from GameObject
  8. GameSetup.cu, GameSetup.cuh - user settings file. Put there scenes creation and work with objects. Function startGame will be called from main()
  9. IComponent.cuh - system base component class. Do not inherit from it! Inherit from Component instead
  10. IMovable.cuh - class to define resource which can be moved from host (CPU) to device (GPU) and vice-versa
  11. ImportMesh.h, ImportMesh.cpp - functions to import mesh data from .obj files
  12. Light.cu, Light.cuh - main light class. It realizes default lightning system
  13. Mesh.cu, Mesh.cuh - Mesh class. It realizes 3D objects and collision of point and Mesh
  14. OldExamples.cuh - some examples of code, which were used before and I'm scared to delete them right now
  15. Quaternion.cu, Quaternion.cuh - realization of mathematical Quaternion object (used for rotating 3D objects)
  16. Scene.cu, Scene.cuh - Scene class. It stores all objects in scene and moves them from host to device when it's needed.
  17. Transform.cu, Transform.cuh - component responsible for movements of a GameObject. It can rotate it, translate or scale.
  18. Triangle.cu, Triangle.cuh - a class for Mesh triangle item. It realizes ray casting (intersection with ray). Also Triangle is immutable
  19. Vector3.cu, Vector3.cuh - realization of mathematical 3D vector object (used for determing positions and directions in 3D world)
  20. VisualEffect.cuh - a basic class for visual effects applied to the pixelmap after rendering (such effect can be blur or smth else)
