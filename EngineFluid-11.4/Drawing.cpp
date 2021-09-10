#include "Drawing.h"
#include <thread>
#include <assert.h>

bool UIDrawer::classRegistered = false;
const wchar_t* UIDrawer::windowClassname = L"EngineFluid UIDrawer window";

UIDrawer::UIDrawer(int w, int h, const char* title , HINSTANCE hInstance) {
	width = w;
	height = h;

    if (!classRegistered) {
        registerClass(hInstance);
    }

    createWindow(hInstance, title);

    colorMap = new Color[(unsigned int)(width * height)];

    objectInitialized = true;
}

UIDrawer::~UIDrawer()
{
    delete[] colorMap;
    objectInitialized = false;
    width = 0;
    height = 0;
    DeleteDC(window);
    window = NULL;
}

void UIDrawer::setPixel(int x, int y, Color color) {
    assert(objectInitialized);

    assert(0 <= y && y < height);
    assert(0 <= x && x < width);

    colorMap[y * width + x] = color;
}

void UIDrawer::processWindowEventsLoop(void(*processor)(MSG*))
{
    MSG msg = { };
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);

        if (processor != NULL) {
            processor(&msg);
        }
        
        DispatchMessage(&msg);
    }
}

void UIDrawer::display()
{
    assert(objectInitialized);

    HBITMAP map = CreateBitmap(width, height, 1, 8 * sizeof(int), (void*)colorMap);

    HDC temp = CreateCompatibleDC(window);
    SelectObject(temp, map);

    BitBlt(window, 0, 0, width, height, temp, 0, 0, SRCCOPY);

    DeleteObject(map);
    DeleteDC(temp);
}

void UIDrawer::clear()
{
    fill(RGB(0, 0, 0));
}

void UIDrawer::fill(Color color)
{
    assert(objectInitialized);

    for (int i = 0; i < width * height; i++)
    {
        colorMap[i] = color;
    }
}

void UIDrawer::resetColorMap(Color* colorMap)
{
    delete[] this->colorMap;

    this->colorMap = colorMap;
}

Color* UIDrawer::getColorMap()
{
    return colorMap;
}

int UIDrawer::getWidth()
{
    return width;
}

int UIDrawer::getHeight()
{
    return height;
}

void UIDrawer::createWindow(HINSTANCE hInstance, const char* title)
{
    // Create the window.

    HWND hwnd = CreateWindowEx(0, (PCSTR)windowClassname, (LPCSTR)title, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, width, height,
        NULL, NULL, hInstance, NULL
    );

    if (hwnd == NULL)
    {
        MessageBox(NULL, (LPCSTR)"Cannot create window!", (LPCSTR)"Error!", MB_OK);
        return;
    }

    ShowWindow(hwnd, SW_SHOW);

    this->window = GetDC(hwnd);
}

void UIDrawer::registerClass(HINSTANCE hInstance)
{
    assert(!classRegistered);

    // Register the window class.
    const wchar_t *CLASS_NAME = (const wchar_t*)windowClassname;

    WNDCLASS wc = { };

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = (LPCSTR)CLASS_NAME;

    RegisterClass(&wc);

    classRegistered = true;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}
