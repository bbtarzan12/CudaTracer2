#include "CPURenderer.h"
#include "ShaderCommon.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

CPURenderer::CPURenderer()
{

}

CPURenderer::~CPURenderer()
{
	cout << "[Renderer] CPURenderer Released" << endl;
	delete currentOption.data;
}

void CPURenderer::Init(RendererOption option)
{
	cout << "[Renderer] Init CPURenderer" << endl;
	Renderer::Init(option);
	GLFWManager::Init(option.width, option.height, "CPU Tracer", this);
	
	{
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
		glBindTexture(GL_TEXTURE_2D, 0);

		cudaViewProgramID = LoadShaders("cudaView.vert", "cudaView.frag");

		GLfloat vertices[] =
		{
			// positions        // normal          // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 0.0f, -1.0f,  0.0f, 0.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f,  0.0f, 1.0f,
			1.0f,  1.0f, 0.0f,  0.0f, 0.0f, -1.0f,  1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,  0.0f, 0.0f, -1.0f,  1.0f, 1.0f,
		};

		glGenVertexArrays(1, &cudaVAO);
		glGenBuffers(1, &cudaVBO);

		glBindVertexArray(cudaVAO);
		glBindBuffer(GL_ARRAY_BUFFER, cudaVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) 0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (3 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (6 * sizeof(float)));
		glBindVertexArray(0);
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
		spheres.emplace_back(vec3(0, 1040, 0), 1000, 4);
		spheres.emplace_back(vec3(0, -1010, 0), 1000, 4);
		spheres.emplace_back(vec3(1040, 0, 0), 1000, 4);
		spheres.emplace_back(vec3(-1040, 0, 0), 1000, 5);
		spheres.emplace_back(vec3(0, 0, 1040), 1000, 6);
		spheres.emplace_back(vec3(0, 0, -1040), 1000, 4);
		spheres.emplace_back(vec3(20, 0, 14), 8, 2);
		spheres.emplace_back(vec3(-14, 0, -20), 8, 3);

		lights.push_back(new SphereLight(vec3(0, 20, 0), 5, 1));
	}
}

void CPURenderer::SetCamera(CameraOption option)
{
	Renderer::SetCamera(option);
	camera->UpdateScreen(option.width, option.height);
	camera->UpdateCamera(0);
	cout << "[Renderer] Set Camera To Renderer" << endl;
}

void CPURenderer::Start()
{
	cout << "[Renderer] Start CPURenderer" << endl;

	double lastTime = glfwGetTime();
	while (GLFWManager::WindowShouldClose() == 0)
	{
		glfwPollEvents();

		double currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;

		Update(deltaTime);
		Render(deltaTime);
		cout << deltaTime << endl;
	}
}

void CPURenderer::HandleKeyboard(int key, int scancode, int action, int mods)
{
}

void CPURenderer::HandleMouseClick(int button, int action, int mods)
{
	if (GLFWManager::IsMouseDown(GLFW_MOUSE_BUTTON_MIDDLE))
	{
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
		glfwSetInputMode(GLFWManager::GetWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else
		glfwSetInputMode(GLFWManager::GetWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void CPURenderer::HandleMouseMotion(double xPos, double yPos)
{
	if (GLFWManager::IsMouseDown(GLFW_MOUSE_BUTTON_MIDDLE))
	{
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
		vec2 input = vec2(camera->height / 2 - yPos, camera->width / 2 - xPos);
		camera->UpdateCamera(deltaTime, vec2(0), input);
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
	}
}

void CPURenderer::HandleResize(int width, int height)
{
	camera->UpdateScreen(width, height);
	glViewport(0, 0, width, height);
}

void CPURenderer::Render(double deltaTime)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	const unsigned int& width = camera->width;
	const unsigned int& height = camera->height;

	{
		currentOption.data = new float[width*height * 4];
		currentOption.width = width;
		currentOption.height = height;
		if (camera->dirty)
		{
			currentOption.frame = 1;
			currentOption.isAccumulate = true;
			currentOption.previousFrameData = new float[width*height * 4]{ 0 };
			camera->ResetDirty();
		}
		RenderKernel(camera, spheres, lights, currentOption);
		currentOption.frame++;

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, viewGLTexture);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, currentOption.data);
		}
		glUseProgram(cudaViewProgramID);

		glBindVertexArray(cudaVAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		glBindVertexArray(0);
		glUseProgram(0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwMakeContextCurrent(GLFWManager::GetWindow());
	glfwSwapBuffers(GLFWManager::GetWindow());
}

void CPURenderer::Update(double deltaTime)
{
	vec2 input = vec2(GLFWManager::IsKeyDown(GLFW_KEY_W) ? 1 : GLFWManager::IsKeyDown(GLFW_KEY_S) ? -1 : 0, GLFWManager::IsKeyDown(GLFW_KEY_D) ? 1 : GLFWManager::IsKeyDown(GLFW_KEY_A) ? -1 : 0);
	camera->UpdateCamera(deltaTime, input);
}
