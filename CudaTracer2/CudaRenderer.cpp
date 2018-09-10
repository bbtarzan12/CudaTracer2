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
	GLFWManager::Init(option.width, option.height, "Cuda Tracer", this);
	cout << "[Renderer] CudaRenderer Init" << endl;
}

void CudaRenderer::SetCamera(CameraOption option)
{
	Renderer::SetCamera(option);
	cout << "[Renderer] Set Camera To Renderer" << endl;
}

void CudaRenderer::Start()
{
	cout << "[Renderer] Start CudaRenderer" << endl;

	double lastTime = glfwGetTime();
	while (GLFWManager::WindowShouldClose() == 0)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		double currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;


		glClear(GL_COLOR_BUFFER_BIT);
		glfwSwapBuffers(GLFWManager::GetWindow());
		glfwPollEvents();
	}
}

void CudaRenderer::HandleKeyboard(int key, int scancode, int action, int mods)
{

}

void CudaRenderer::HandleMouse(int button, int action, int mods)
{

}

void CudaRenderer::HandleResize(int width, int height)
{

}

