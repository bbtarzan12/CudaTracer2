#include "CudaRenderer.h"
#include "ShaderCommon.h"
#include "PathKernel.cuh"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iostream>

using namespace std;

CudaRenderer::CudaRenderer()
{
}


CudaRenderer::~CudaRenderer()
{
	cout << "[Renderer] CudaRenderer Free" << endl;
	glDeleteProgram(programID);
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
	cudaDeviceReset();
}

void CudaRenderer::Init(RendererOption option)
{
	Renderer::Init(option);
	GLFWManager::Init(option.width, option.height, "Cuda Tracer", this);
	cout << "[Renderer] CudaRenderer Init" << endl;

	{
		// Cuda Opengl Interop
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &viewGLTexture);
		glBindTexture(GL_TEXTURE_2D, viewGLTexture);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, option.width, option.height, 0, GL_RGBA, GL_FLOAT, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		}
		cudaGraphicsGLRegisterImage(&viewResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		glBindTexture(GL_TEXTURE_2D, 0);

		programID = LoadShaders("vertShader.vert", "fragShader.frag");

		GLfloat vertices[] =
		{
			// positions        // normal          // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 0.0f, -1.0f,  0.0f, 0.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f,  0.0f, 1.0f,
			1.0f,  1.0f, 0.0f,  0.0f, 0.0f, -1.0f,  1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,  0.0f, 0.0f, -1.0f,  1.0f, 1.0f,
		};

		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);

		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) 0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (3 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (6 * sizeof(float)));
	}

	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void) io;
		io.WantCaptureKeyboard = true;
		io.WantCaptureMouse = true;
		ImGui_ImplGlfw_InitForOpenGL(GLFWManager::GetWindow(), false);
		ImGui_ImplOpenGL3_Init("#version 130");
		ImGui::StyleColorsDark();
	}

	{
		// Scene
		materials.push_back(Material());
		materials.push_back(Material(DIFF, vec3(1), vec3(2.2f, 2.2f, 2.2f)));
		materials.push_back(Material(SPEC));
		materials.push_back(Material(TRANS));
		materials.push_back(Material(DIFF, vec3(0.75f, 0.75f, 0.75f)));
		materials.push_back(Material(DIFF, vec3(0.25f, 0.25f, 0.75f)));
		materials.push_back(Material(DIFF, vec3(0.75f, 0.25f, 0.25f)));

		spheres.push_back(Sphere(vec3(0, 1040, 0), 1000, 1));
		spheres.push_back(Sphere(vec3(0, -1010, 0),1000, 4));
		spheres.push_back(Sphere(vec3(1040, 0, 0), 1000, 4));
		spheres.push_back(Sphere(vec3(-1040, 0, 0), 1000, 5));
		spheres.push_back(Sphere(vec3(0, 0, 1040), 1000, 6));
		spheres.push_back(Sphere(vec3(0, 0, -1040), 1000, 4));
		spheres.push_back(Sphere(vec3(20, 0, 14), 8, 2));
		spheres.push_back(Sphere(vec3(-14, 0, -20), 8, 3));
	}
}

void CudaRenderer::SetCamera(CameraOption option)
{
	Renderer::SetCamera(option);
	camera->UpdateScreen(option.width, option.height);
	camera->UpdateCamera(0);
	cout << "[Renderer] Set Camera To Renderer" << endl;
}

void CudaRenderer::Start()
{
	cout << "[Renderer] Start CudaRenderer" << endl;

	float lastTime = glfwGetTime();
	while (GLFWManager::WindowShouldClose() == 0)
	{
		glfwPollEvents();

		float currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;

		Update(deltaTime);
		Render();
	}
}

void CudaRenderer::Update(float deltaTime)
{
	vec2 input = vec2(GLFWManager::IsKeyDown(GLFW_KEY_W) ? 1 : GLFWManager::IsKeyDown(GLFW_KEY_S) ? -1 : 0, GLFWManager::IsKeyDown(GLFW_KEY_D) ? 1 : GLFWManager::IsKeyDown(GLFW_KEY_A) ? -1 : 0);
	camera->UpdateCamera(deltaTime, input);
}

void CudaRenderer::Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	int width = camera->width;
	int height = camera->height;

	{
		// OpenGL + Cuda
		if (toggleCudaView)
		{
			gpuErrorCheck(cudaGraphicsMapResources(1, &viewResource));
			gpuErrorCheck(cudaGraphicsSubResourceGetMappedArray(&viewArray, viewResource, 0, 0));

			cudaResourceDesc viewCudaArrayResourceDesc;
			{
				viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
				viewCudaArrayResourceDesc.res.array.array = viewArray;
			}

			if (camera->dirty)
			{
				frame = 1;
				camera->ResetDirty();
			}

			RenderOption option;
			option.frame = frame;
			option.enableDof = false;
			option.loopX = 1;
			option.loopY = 1;
			cudaSurfaceObject_t viewCudaSurfaceObject;
			gpuErrorCheck(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
			{
				RenderKernel(camera, spheres, materials, option, viewCudaSurfaceObject);
			}
			gpuErrorCheck(cudaDestroySurfaceObject(viewCudaSurfaceObject));
			gpuErrorCheck(cudaGraphicsUnmapResources(1, &viewResource));
			cudaStreamSynchronize(0);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, viewGLTexture);

			glUseProgram(programID);

			glBindVertexArray(VAO);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
			glBindVertexArray(0);
			frame++;
		}
		else
		{

		}
	}

	{
		// Imgui
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		static float f = 0.0f;
		static int counter = 0;

		ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

		ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)

		ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f

		if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
			counter++;
		ImGui::SameLine();
		ImGui::Text("counter = %d", counter);

		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::End();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
	glfwMakeContextCurrent(GLFWManager::GetWindow());
	glfwSwapBuffers(GLFWManager::GetWindow());
}

void CudaRenderer::HandleKeyboard(int key, int scancode, int action, int mods)
{
	if(GLFWManager::IsKeyDown(GLFW_KEY_ESCAPE))
		glfwSetWindowShouldClose(GLFWManager::GetWindow(), GLFW_TRUE);
	if (GLFWManager::IsKeyDown(GLFW_KEY_Q))
		toggleCudaView = !toggleCudaView;
}

void CudaRenderer::HandleMouseClick(int button, int action, int mods)
{
	if (GLFWManager::IsMouseDown(GLFW_MOUSE_BUTTON_MIDDLE))
	{
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
		glfwSetInputMode(GLFWManager::GetWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else
		glfwSetInputMode(GLFWManager::GetWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void CudaRenderer::HandleMouseMotion(double xPos, double yPos)
{
	if (GLFWManager::IsMouseDown(GLFW_MOUSE_BUTTON_MIDDLE))
	{
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
		vec2 input = vec2(camera->height / 2 - yPos, camera->width / 2 - xPos);
		camera->UpdateCamera(deltaTime, vec2(0), input);
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
	}
}

void CudaRenderer::HandleResize(int width, int height)
{
	camera->UpdateScreen(width, height);
	glViewport(0, 0, width, height);
	cudaGraphicsUnregisterResource(viewResource);
	glBindTexture(GL_TEXTURE_2D, viewGLTexture);
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	cudaGraphicsGLRegisterImage(&viewResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

