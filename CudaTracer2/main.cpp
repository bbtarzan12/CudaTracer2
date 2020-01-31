//#include "CPURenderer.h"
#include "CudaRenderer.h"


int main(int argc, char **argv)
{
	unique_ptr<Renderer> renderer = make_unique<CudaRenderer>();

	RendererOption rendererOption;
	rendererOption.width = 1280/2;
	rendererOption.height = 720/2;

	CameraOption cameraOption;
	cameraOption.width = rendererOption.width;
	cameraOption.height = rendererOption.height;
	cameraOption.position = vec3(-14.0694f, -0.301702f, 30.0197f);
	cameraOption.fov = 70.0f;
	cameraOption.nearPlane = 0.1f;
	cameraOption.farPlane = 1000.0f;
	cameraOption.moveSpeed = 25.0f;
	cameraOption.mouseSpeed = 10.0f;
	cameraOption.pitch = -13.8f;
	cameraOption.yaw = 157.067f;
	cameraOption.aperture = 0;
	cameraOption.focalDistance = 0.1f;

	renderer->Init(rendererOption);;
	renderer->SetCamera(cameraOption);
	renderer->Start();

	// Cleanup
	glfwTerminate();
	return 0;
}