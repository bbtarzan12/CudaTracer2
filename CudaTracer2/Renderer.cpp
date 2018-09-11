#include "Renderer.h"

void Renderer::Init(RendererOption option)
{
	this->rendererOption = option;
}

void Renderer::SetCamera(CameraOption option)
{
	camera = make_shared<Camera>(option);
}