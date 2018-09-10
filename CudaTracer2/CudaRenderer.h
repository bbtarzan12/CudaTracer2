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

private:

};

#endif