#include "Renderer.h"

void Renderer::Init(RendererOption option)
{
	this->rendererOption = option;

	glfwSetErrorCallback(ErrorCallback);
	if (!glfwInit())
	{
		cerr << "[Error] GLFW 초기화 실패" << endl;
		exit(EXIT_FAILURE);
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 4);
	window = glfwCreateWindow(option.widht, option.height, "Cuda Tracer", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glClear(GL_COLOR_BUFFER_BIT);

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GL_TRUE);
	glfwSetKeyCallback(window, KeyboardCallback);
	glfwSetMouseButtonCallback(window, MouseCallback);
	glfwSetWindowSizeCallback(window, ResizeCallback);

	glewExperimental = GL_TRUE;

	GLenum errorCode = glewInit();
	if (GLEW_OK != errorCode)
	{
		cerr << "[Error] GLEW 초기화 실패" << glewGetErrorString(errorCode) << endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	if (!GLEW_VERSION_3_3)
	{
		cerr << "[Error] 3.3 API가 유효하지 않습니다" << endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	cout << "[OpenGL] OpenGL version: " << glGetString(GL_VERSION) << endl;
	cout << "[OpenGL] GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	cout << "[OpenGL] Vendor: " << glGetString(GL_VENDOR) << endl;
	cout << "[OpenGL] Renderer: " << glGetString(GL_RENDERER) << endl;

	glfwSwapInterval(1); // vSync
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
}

void Renderer::SetCamera(CameraOption option)
{
	camera = make_unique<Camera>(option);
}

bool Renderer::UpdateRenderer()
{
	float lastTime = glfwGetTime();
	while (glfwWindowShouldClose(window) == 0)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		float currentTime = glfwGetTime();
		float deltaTime = currentTime - lastTime;
		lastTime = currentTime;


		glClear(GL_COLOR_BUFFER_BIT);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	return false;
}

void Renderer::KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void Renderer::ResizeCallback(GLFWwindow* window, int width, int height)
{

}

void Renderer::MouseCallback(GLFWwindow * window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		if (action == GLFW_PRESS)
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		if (action == GLFW_RELEASE)
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void Renderer::ErrorCallback(int errorCode, const char* errorDescription)
{
	fprintf(stderr, "Error: %s\n", errorDescription);
}
