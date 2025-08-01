#include <raylib.h>
#include <raymath.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define RES 512

static long count_lines(const char* path){
    FILE *f=fopen(path,"r");
    if(!f) return 0;
    long n=0; int c;
    while((c=fgetc(f))!=EOF) if(c=='\n') n++;
    fclose(f);
    return n;
}

static float* load_heightmap(const char* dataset,long idx){
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),"python3 render/eval_height.py %s %ld %d",dataset,idx,RES);
    FILE *p=popen(cmd,"r");
    if(!p) return NULL;
    size_t total=RES*RES;
    float *h=malloc(total*sizeof(float));
    fread(h,sizeof(float),total,p);
    pclose(p);
    return h;
}

static void camera_update(Camera3D* cam){
    float dt=GetFrameTime();
    float base=60.0f;
    float speed=IsKeyDown(KEY_LEFT_SHIFT)?base*4.0f:base;
    Vector3 forward=Vector3Normalize(Vector3Subtract(cam->target,cam->position));
    Vector3 right=Vector3Normalize(Vector3CrossProduct(forward,cam->up));
    Vector3 move={0};
    if(IsKeyDown(KEY_W)) move=Vector3Add(move,Vector3Scale(forward,speed*dt));
    if(IsKeyDown(KEY_S)) move=Vector3Add(move,Vector3Scale(forward,-speed*dt));
    if(IsKeyDown(KEY_A)) move=Vector3Add(move,Vector3Scale(right,-speed*dt));
    if(IsKeyDown(KEY_D)) move=Vector3Add(move,Vector3Scale(right,speed*dt));
    cam->position=Vector3Add(cam->position,move);
    cam->target=Vector3Add(cam->target,move);
    Vector2 md=GetMouseDelta();
    float sens=0.00003f;
    float yaw=-md.x*sens;
    float pitch=-md.y*sens;
    Matrix rotYaw=MatrixRotate(cam->up,yaw);
    Vector3 dir=Vector3Transform(forward,rotYaw);
    Vector3 newRight=Vector3Normalize(Vector3CrossProduct(dir,cam->up));
    Matrix rotPitch=MatrixRotate(newRight,pitch);
    dir=Vector3Transform(dir,rotPitch);
    cam->target=Vector3Add(cam->position,dir);
}

int main(int argc,char** argv){
    const char* dataset=argc>1?argv[1]:"data/raw/demo/dataset_all.jsonl";
    long idx=argc>2?atol(argv[2]):-1;
    if(idx<0){
        long lines=count_lines(dataset);
        srand(time(NULL));
        idx=rand()%lines;
    }
    float* height=load_heightmap(dataset,idx);
    if(!height) return 1;
    SetConfigFlags(FLAG_MSAA_4X_HINT|FLAG_VSYNC_HINT);
    InitWindow(1280,720,"text2terrain");
    SetTargetFPS(60);
    Camera3D cam={0};
    cam.position=(Vector3){0,200,-400};
    cam.target=(Vector3){0,0,0};
    cam.up=(Vector3){0,1,0};
    cam.fovy=45;
    cam.projection=CAMERA_PERSPECTIVE;
    DisableCursor();
    while(!WindowShouldClose()){
        camera_update(&cam);
        BeginDrawing();
        ClearBackground(BLACK);
        BeginMode3D(cam);
        for(int y=0;y<RES;y++){
            for(int x=0;x<RES;x++){
                int k=y*RES+x;
                Vector3 p={(float)x-RES/2.0f,height[k],(float)y-RES/2.0f};
                DrawPoint3D(p,WHITE);
            }
        }
        EndMode3D();
        EndDrawing();
    }
    CloseWindow();
    free(height);
    return 0;
}