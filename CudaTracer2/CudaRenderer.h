#ifndef CUDARENDERER_H
#define CUDARENDERER_H

#include "Renderer.h"

class CudaRenderer : public Renderer
{
public:
	CudaRenderer();
	~CudaRenderer() override;

	void Init(RendererOption option) override;
	void SetCamera(CameraOption option) override;
	void Start() override;

	// Callbacks
	static void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void ResizeCallback(GLFWwindow* window, int width, int height);
	static void MouseCallback(GLFWwindow* window, int button, int action, int mods);
	static void ErrorCallback(int errorCode, const char* errorDescription);

private:
	GLFWwindow* window;
	float deltaTime;

};

#endif