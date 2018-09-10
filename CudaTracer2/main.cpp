#include "CudaRenderer.h"


int main(int argc, char **argv)
{

	unique_ptr<Renderer> renderer = make_unique<CudaRenderer>();

	RendererOption rendererOption;
	rendererOption.widht = 1280;
	rendererOption.height = 720;
	rendererOption.type = RendererType::CUDA;

	CameraOption cameraOption;
	cameraOption.position = vec3(-18.192522f, 6.537864f, 31.417776f);
	cameraOption.fov = 70.0f;
	cameraOption.nearPlane = 0.1f;
	cameraOption.farPlane = 1000.0f;
	cameraOption.moveSpeed = 25.0f;
	cameraOption.mouseSpeed = 10.0f;
	cameraOption.pitch = -22.780003f;
	cameraOption.yaw = 155.000137f;
	cameraOption.aperture = 0;
	cameraOption.focalDistance = 0.1f;

	renderer->Init(rendererOption);;
	renderer->SetCamera(cameraOption);
	renderer->Start();

	// Cleanup
	cudaDeviceReset();
	glfwTerminate();
	return 0;
}