#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aOffset;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform mat3 NormalMatrix;

uniform float PointSize;

void main()
{
    //FragPos = vec3(model * vec4(aPos, 1.0));
    FragPos = vec3(model * vec4(PointSize * aPos + aOffset, 1.0));
    Normal = NormalMatrix * aNormal;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}