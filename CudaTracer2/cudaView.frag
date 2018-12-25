#version 330 core

in vec2 outUV;

out vec3 color;

uniform sampler2D map;

void main()
{
	color = texture(map, outUV).rgb;
}