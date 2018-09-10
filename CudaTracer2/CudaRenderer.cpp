#include "CudaRenderer.h"
#include <iostream>

using namespace std;

CudaRenderer::CudaRenderer()
{
}


CudaRenderer::~CudaRenderer()
{
	cout << "[Renderer] CudaRenderer Free" << endl;
}

void CudaRenderer::Init(RendererOption option)
{
	Renderer::Init(option);
	cout << "[Renderer] CudaRenderer Init" << endl;
}

void CudaRenderer::SetCamera(CameraOption option)
{
	Renderer::SetCamera(option);
	cout << "[Renderer] Set Camera To Renderer" << endl;
}