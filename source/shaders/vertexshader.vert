#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 viewPos;

uniform float pointRadius;


void main()
{
    vec4 pointInWorld = model * vec4(aPos, 1.0f); 
    gl_Position = projection * view * pointInWorld;


    float dist = length(viewPos - vec3(pointInWorld));

    gl_PointSize = pointRadius*100 / dist;
}