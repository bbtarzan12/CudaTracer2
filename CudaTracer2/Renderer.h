#ifndef RENDERER_H
#define RENDERER_H

#include "TracingCommon.h"

enum class RendererType { CUDA };

struct RendererOption
{
	RendererType type;
	int widht, height;
};

struct RenderOption
{

};

class Renderer
{
public:
	virtual ~Renderer() = default;
	virtual void Init(RendererOption option);
	virtual void SetCamera(CameraOption option);
	virtual bool UpdateRenderer();

	// Callbacks
	static void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void ResizeCallback(GLFWwindow* window, int width, int height);
	static void MouseCallback(GLFWwindow* window, int button, int action, int mods);
	static void ErrorCallback(int errorCode, const char* errorDescription);

private:
	GLFWwindow* window;
	RendererOption rendererOption;
	unique_ptr<Camera> camera;
	float deltaTime;
};

#endif