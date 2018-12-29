#version 330 core

out vec4 FragColor;

in vec3 FragPos;  
in vec3 Normal;  
in vec2 TexCoords;

uniform vec3 camPos;
uniform vec3 sunDir;
uniform float metalic;
uniform float sunPower;

void main()
{
    // ambient
    vec3 ambient = vec3(0.1, 0.2, 0.3);
  	
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-sunDir);  
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = /*light.diffuse */ sunPower * diff * vec3(1);  
    
    // specular
    vec3 viewDir = normalize(camPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), metalic);
    vec3 specular = /*light.specular */ spec * vec3(1);  
        
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}