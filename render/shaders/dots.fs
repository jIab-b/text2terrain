#version 330 core
in vec3 fragPos;
in vec2 fragTexCoord;

uniform float dotScale = 16.0;

out vec4 fragColor;

void main(){
    vec2 grid = fract(fragTexCoord * dotScale);
    float d = step(max(grid.x, grid.y), 0.95);
    vec3 base = vec3(0.8);
    fragColor = vec4(mix(base, vec3(0.0), d), 1.0);
}