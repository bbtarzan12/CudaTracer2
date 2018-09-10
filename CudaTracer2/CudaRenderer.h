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
	void HandleKeyboard(int key, int scancode, int action, int mods) override;
	void HandleMouse(int button, int action, int mods) override;
	void HandleResize(int width, int height) override;

private:
	double deltaTime;

};

#endif