#ifndef DRAWING
#define DRAWING
#include <Windows.h>

typedef COLORREF Color;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

class UIDrawer {
public:
	UIDrawer(int w, int h, const char* title, HINSTANCE hInstance);
	~UIDrawer();
	void setPixel(int x, int y, Color color);
	void processWindowEventsLoop(void(*processor)(MSG*) = NULL);
	void display();
	void clear();
	void fill(Color color);
	void resetColorMap(Color* colorMap);
	Color* getColorMap();
	int getWidth();
	int getHeight();
private:
	void createWindow(HINSTANCE hInstance, const char* title);
	void registerClass(HINSTANCE hInstance);

	HDC window;
	int width;
	int height;
	Color* colorMap;
	bool objectInitialized = false;

	static bool classRegistered;
	const static wchar_t* windowClassname;
};


#endif
