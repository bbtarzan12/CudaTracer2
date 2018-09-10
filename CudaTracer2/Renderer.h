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
	virtual void Start() = 0;

private:
	RendererOption rendererOption;
	unique_ptr<Camera> camera;
};

#endif