#include "raylib.h"
#include "raymath.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static Camera3D init_camera(int tile_size){
    Camera3D cam={0};
    float half_tile=(float)tile_size*0.5f;
    cam.position=(Vector3){half_tile,60,half_tile+100};
    cam.target=(Vector3){half_tile,0,half_tile};
    cam.up=(Vector3){0,1,0};
    cam.fovy=45;
    cam.projection=CAMERA_PERSPECTIVE;
    return cam;
}

static bool cursor_disabled = true;

static void handle_input(Camera3D *cam){
    if(IsKeyPressed(KEY_TAB)){
        if(cursor_disabled){
            EnableCursor();
            cursor_disabled = false;
        } else {
            DisableCursor();
            cursor_disabled = true;
        }
    }
    if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !cursor_disabled){
        DisableCursor();
        cursor_disabled = true;
    }
}

int main(int argc,char **argv){
    if(argc<2){
        printf("usage: %s <sample_json>\n",argv[0]);
        return 1;
    }
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),"python3 render/reconstruct_heightmap.py %s /tmp/hm.png",argv[1]);
    if(system(cmd)!=0){
        fprintf(stderr,"reconstruct script failed\n");
        return 1;
    }

    Image img=LoadImage("/tmp/hm.png");
    if(img.data==NULL){
        fprintf(stderr,"Failed to load heightmap\n");
        return 1;
    }
    printf("Loaded heightmap: %dx%d, format=%d\n",img.width,img.height,img.format);

    InitWindow(1024,768,"Terrain viewer");
    SetExitKey(KEY_NULL);
    SetTargetFPS(60);
    cursor_disabled = false;

    int tile=img.width;
    Mesh mesh=GenMeshHeightmap(img,(Vector3){(float)tile,60.0f,(float)tile});
    printf("Generated mesh: %d vertices, %d triangles\n",mesh.vertexCount,mesh.triangleCount);
    Model model=LoadModelFromMesh(mesh);
    Texture2D tex=LoadTextureFromImage(img);
    UnloadImage(img);
    model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture=tex;

    Shader sh=LoadShader(NULL,"render/shaders/dots.fs");
    if(sh.id!=0){
        int loc=GetShaderLocation(sh,"dotScale");
        float scale=16.0f; SetShaderValue(sh,loc,&scale,SHADER_UNIFORM_FLOAT);
        model.materials[0].shader=sh;
    }

    Camera3D cam=init_camera(tile);

    while(!WindowShouldClose()){
        handle_input(&cam);
        
        if(cursor_disabled){
            UpdateCamera(&cam,CAMERA_FIRST_PERSON);
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(cam);
        DrawModel(model,(Vector3){0,0,0},1.0f,WHITE);
        EndMode3D();
        DrawFPS(10,10);
        if(!cursor_disabled){
            DrawText("TAB: Toggle mouse look | Click: Enable mouse look",10,30,20,BLACK);
        }
        EndDrawing();
    }
    UnloadShader(sh);
    UnloadTexture(tex);
    UnloadModel(model);
    CloseWindow();
    return 0;
}