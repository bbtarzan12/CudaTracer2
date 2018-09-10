#ifndef RENDERER_H
#define RENDERER_H

#include "GLFWManager.h"
#include "TracingCommon.h"

enum class RendererType { CUDA };

struct RendererOption
{
	RendererType type;
	int width, height;
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
	virtual void Start() = 0;
	virtual void HandleKeyboard(int key, int scancode, int action, int mods) = 0;
	virtual void HandleMouse(int button, int action, int mods) = 0;
	virtual void HandleResize(int width, int height) = 0;

private:
	RendererOption rendererOption;
	unique_ptr<Camera> camera;
};

#endif