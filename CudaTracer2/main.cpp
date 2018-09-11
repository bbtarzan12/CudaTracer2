#include "CudaRenderer.h"

int main(int argc, char **argv)
{

	unique_ptr<Renderer> renderer = make_unique<CudaRenderer>();

	RendererOption rendererOption;
	rendererOption.width = 1280;
	rendererOption.height = 720;
	rendererOption.type = RendererType::CUDA;

	CameraOption cameraOption;
	cameraOption.width = rendererOption.width;
	cameraOption.height = rendererOption.height;
	cameraOption.position = vec3(4.0f, 3.0f, -3.0f);
	cameraOption.fov = 70.0f;
	cameraOption.nearPlane = 0.1f;
	cameraOption.farPlane = 1000.0f;
	cameraOption.moveSpeed = 25.0f;
	cameraOption.mouseSpeed = 10.0f;
	cameraOption.pitch = -34.529;
	cameraOption.yaw = 319.545f;
	cameraOption.aperture = 0;
	cameraOption.focalDistance = 0.1f;

	renderer->Init(rendererOption);;
	renderer->SetCamera(cameraOption);
	renderer->Start();

	// Cleanup
	glfwTerminate();
	return 0;
}