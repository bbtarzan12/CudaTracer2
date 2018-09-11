#ifndef H_RENDERER
#define H_RENDERER

#include "GLFWManager.h"
#include "TracingCommon.h"

enum class RendererType { CUDA };

struct RendererOption
{
	RendererType type;
	int width, height;
};

class Renderer
{
public:
	virtual ~Renderer() = default;
	virtual void Init(RendererOption option);
	virtual void SetCamera(CameraOption option);
	virtual void Start() = 0;
	virtual void HandleKeyboard(int key, int scancode, int action, int mods) = 0;
	virtual void HandleMouseClick(int button, int action, int mods) = 0;
	virtual void HandleMouseMotion(double xPos, double yPos) = 0;
	virtual void HandleResize(int width, int height) = 0;

protected:
	RendererOption rendererOption;
	shared_ptr<Camera> camera;
};

#endif