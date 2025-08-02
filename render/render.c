#include <raylib.h>
#include <raymath.h>
#include <stdbool.h>
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

static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}


static float* load_heightmap(const char* dataset,long idx){
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),"python3 render/eval_height.py %s %ld %d",dataset,idx,RES);
    FILE *p=popen(cmd,"r");
    if(!p) return NULL;
    size_t total=RES*RES;
    float *h=malloc(total*sizeof(float));
    size_t got = fread(h, sizeof(float), total, p);
    printf("Debug: load_heightmap expected %zu floats, got %zu\n", total, got);
    pclose(p);
    return h;
}



static void first_person_control(Camera *camera) {
    static float camera_azimuth = 0.0f;
    static float camera_elevation = 0.0f;
    static bool initialized = false;
    
    if (!initialized) {
        Vector3 dir = Vector3Subtract(camera->target, camera->position);
        dir = Vector3Normalize(dir);
        camera_azimuth = atan2f(dir.z, dir.x);
        camera_elevation = asinf(dir.y);
        initialized = true;
    }
    
    Vector2 center = {GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
    Vector2 current = GetMousePosition();
    Vector2 md = Vector2Subtract(current, center);
    if (Vector2LengthSqr(md) > 0.01f) {
        float sens = 0.0002f;
        camera_azimuth += md.x * sens;
        camera_elevation -= md.y * sens;
        camera_elevation = clampf(camera_elevation, -PI/2.0f + 0.1f, PI/2.0f - 0.1f);
    }
    SetMousePosition(center.x, center.y);
    Vector3 front = {cosf(camera_elevation) * cosf(camera_azimuth), sinf(camera_elevation), cosf(camera_elevation) * sinf(camera_azimuth)};
    front = Vector3Normalize(front);
    float speed = 0.1f;
    if (IsKeyDown(KEY_LEFT_SHIFT)) speed *= 10.0f;
    if (IsKeyDown(KEY_W)) camera->position = Vector3Add(camera->position, Vector3Scale(front, speed));
    if (IsKeyDown(KEY_S)) camera->position = Vector3Subtract(camera->position, Vector3Scale(front, speed));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(front, (Vector3){0,1,0}));
    if (IsKeyDown(KEY_D)) camera->position = Vector3Add(camera->position, Vector3Scale(right, speed));
    if (IsKeyDown(KEY_A)) camera->position = Vector3Subtract(camera->position, Vector3Scale(right, speed));
    if (IsKeyDown(KEY_SPACE)) camera->position.y += speed;
    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) camera->position.y -= speed;
    camera->target = Vector3Add(camera->position, front);
}





void init_camera(Camera* camera) {

    camera->position = (Vector3){ 0.2f, 0.4f, 0.2f };    // Camera position
    camera->target = (Vector3){ 0.185f, 0.4f, 0.0f };    // Camera looking at point
    camera->up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera->fovy = 45.0f;                                // Camera field-of-view Y
    camera->projection = CAMERA_PERSPECTIVE;             // Camera projection type

}



int main(int argc,char** argv){
    const char* dataset = "data/raw/demo/dataset_all.jsonl";
    long idx = -1;
    if(argc>1){
        char* end; long val=strtol(argv[1],&end,10);
        if(*end=='\0') idx = val;        // numeric => index
        else            dataset = argv[1]; // otherwise dataset path
    }
    if(argc>2){
        char* end; long val=strtol(argv[2],&end,10);
        if(*end=='\0') idx = val;
    }
    long lines=count_lines(dataset);
    if(lines==0) return 1;
    if(idx<0 || idx>=lines){
        srand(time(NULL));
        idx = rand()%lines;
    }
    float* height=load_heightmap(dataset,idx);
    if(!height) return 1;
    SetConfigFlags(FLAG_MSAA_4X_HINT|FLAG_VSYNC_HINT);
    InitWindow(1280,720,"text2terrain");
    SetTargetFPS(60);

    Camera cam = {0};
    init_camera(&cam);
    //DisableCursor();
    
    float minh = height[0], maxh = height[0];
    for(int i = 1; i < RES*RES; i++){
        if(height[i] < minh) minh = height[i];
        if(height[i] > maxh) maxh = height[i];
    }
    float rangeh = (maxh - minh) > 1e-6f ? (maxh - minh) : 1.0f;
    float centerY = minh + rangeh * 0.5f;

    while(!WindowShouldClose()){


        Vector3 oldCamPos = cam.position;
        //UpdateCamera(&cam, CAMERA_FIRST_PERSON);
        BeginDrawing();
        ClearBackground(BLACK);
        first_person_control(&cam);
        BeginMode3D(cam);
        for(int y = 0; y < RES; y++){
            for(int x = 0; x < RES; x++){
                int k = y*RES + x;
                float h = height[k];
                unsigned char c = (unsigned char)(((h - minh)/rangeh)*255.0f);
                Color col = (Color){ c, c, c, 255 };
                Vector3 p = { (float)x - RES/2.0f, h, (float)y - RES/2.0f };
                DrawPoint3D(p, col);
            }
        }
        EndMode3D();
        EndDrawing();
    }
    CloseWindow();
    free(height);
    return 0;
}
