#include "Light.cuh"


EngineColor::EngineColor()
{
	r = 0;
	g = 0;
	b = 0;
	a = 1;
}

EngineColor::EngineColor(_byte r, _byte g, _byte b, _byte a)
{
	this->r = r;
	this->g = g;
	this->b = b;
	this->a = a;
}

EngineColor::EngineColor(Color base)
{
	// windows.h COLORREF (a.k.a. Color there) is actually unsigned long long
	// so it stores data like this:
	// | 40 bites empty | 8 bites blue color | 8 bites green color | 8 bites red color |
	unsigned long colorUnwrapped = base;
	r = colorUnwrapped % 256;
	g = (colorUnwrapped >> 8) % 256;
	b = (colorUnwrapped >> 16) % 256;
	a = 1;
}

EngineColor EngineColor::operator=(const EngineColor& other)
{
	r = other.r;
	g = other.g;
	b = other.b;
	a = other.a;

	return *this;
}

const EngineColor EngineColor::operator+(const EngineColor& other) const
{
	int _r = (int)r + other.r;
	int _b = (int)b + other.b;
	int _g = (int)g + other.g;
	int _a = (int)a + other.a;
	_a = max(1, _a);

	return EngineColor((_r * 255) / _a, (_g * 255) / _a, (_b * 255) / _a, min(a, 255));
}

const EngineColor EngineColor::operator==(const EngineColor& other) const
{
	return r == other.r &&
		g == other.g &&
		b == other.b && 
		a == other.a;
}

Color EngineColor::toWinColor()
{
	return RGB(r, g, b);
}


const int GeneralLight::typeId() const
{
	return GENERAL_LIGHT_TYPEID;
}

GeneralLight::GeneralLight(GameObject* parent, EngineColor lightColor) : Component(parent)
{
	_lightColor = lightColor;
}

void GeneralLight::awake()
{
	_transform = gameObject()->getComponentOfType<Transform>();
}

EngineColor GeneralLight::getLight(Vector3 normal)
{
	int angle_parameter = max(normal.angle_cos(_transform->forward()), 0) * 255;

	EngineColor e = _lightColor;
	e.a = angle_parameter;
	e = e + EngineColor(0, 0, 0, 1);
	return e;
}

void GeneralLight::moveToDevice()
{
}

void GeneralLight::moveToHost()
{
}

void GeneralLight::resetGameObject(GameObject* object)
{
	Component::resetGameObject(object);
	_transform = object->getComponentOfType<Transform>();
}
