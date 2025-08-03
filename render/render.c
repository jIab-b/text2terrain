#include <raylib.h>
#include <raymath.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define RES 512

static char temp_file[256] = {0};

static void cleanup_temp_file() {
    if (temp_file[0] != '\0') {
        unlink(temp_file);
    }
}

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
    snprintf(temp_file, sizeof(temp_file), "/tmp/heightmap_%d_%ld.bin", getpid(), idx);
    atexit(cleanup_temp_file);
    
    char cmd[1024];
    snprintf(cmd,sizeof(cmd),"python3 render/eval_height.py %s %ld %d %s 2>/dev/null",dataset,idx,RES,temp_file);
    printf("Executing: %s\n", cmd);
    
    int exit_code = system(cmd);
    if (exit_code != 0) {
        fprintf(stderr, "Error: heightmap generation failed\n");
        return NULL;
    }
    
    FILE *f = fopen(temp_file, "rb");
    if (!f) {
        fprintf(stderr, "Error: failed to open temp file\n");
        return NULL;
    }
    
    size_t total=RES*RES;
    float *h=malloc(total*sizeof(float));
    if(!h) {
        fprintf(stderr, "Error: failed to allocate memory for heightmap\n");
        fclose(f);
        return NULL;
    }
    
    size_t got = fread(h, sizeof(float), total, f);
    fclose(f);
    
    if (got != total) {
        free(h);
        return NULL;
    }
    
    return h;
}



static void first_person_control(Camera *camera) {
    static float yaw = 0.0f;
    static float pitch = 0.0f;
    static bool initialized = false;
    if (!initialized) {
        //DisableCursor();
        HideCursor();
        Vector3 dir = Vector3Normalize(Vector3Subtract(camera->target, camera->position));
        yaw = atan2f(dir.z, dir.x);
        pitch = asinf(dir.y);
        initialized = true;
    }
    float dt = GetFrameTime();
    Vector2 md = GetMouseDelta();
    float sens = 0.002f;
    yaw += md.x * sens;
    pitch -= md.y * sens;
    pitch = clampf(pitch, -PI/2.0f + 0.1f, PI/2.0f - 0.1f);
    Vector3 front = {cosf(pitch) * cosf(yaw), sinf(pitch), cosf(pitch) * sinf(yaw)};
    Vector3 right = Vector3Normalize(Vector3CrossProduct(front, (Vector3){0,1,0}));
    float baseSpeed = 4.0f;
    float speed = baseSpeed * dt * (IsKeyDown(KEY_LEFT_SHIFT) ? 10.0f : 1.0f);
    if (IsKeyDown(KEY_W)) camera->position = Vector3Add(camera->position, Vector3Scale(front, speed));
    if (IsKeyDown(KEY_S)) camera->position = Vector3Subtract(camera->position, Vector3Scale(front, speed));
    if (IsKeyDown(KEY_D)) camera->position = Vector3Add(camera->position, Vector3Scale(right, speed));
    if (IsKeyDown(KEY_A)) camera->position = Vector3Subtract(camera->position, Vector3Scale(right, speed));
    if (IsKeyDown(KEY_SPACE)) camera->position.y += speed;
    if (IsKeyDown(KEY_LEFT_CONTROL) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) camera->position.y -= speed;
    camera->target = Vector3Add(camera->position, front);
}




void init_camera(Camera* camera) {

    camera->position = (Vector3){ 0.5f, 0.2f, 0.5f };    // Camera position
    camera->target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at center
    camera->up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector
    camera->fovy = 60.0f;                                // Camera field-of-view Y
    camera->projection = CAMERA_PERSPECTIVE;             // Camera projection type

}



int main(int argc,char** argv){
    const char* dataset = "data/samples/dataset_all.jsonl";
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
    printf("Loading heightmap from dataset %s, index %ld...\n", dataset, idx);
    float* height=load_heightmap(dataset,idx);
    if(!height) {
        fprintf(stderr, "Failed to load heightmap. Exiting.\n");
        return 1;
    }
    printf("Heightmap loaded successfully.\n");
    
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(1280,720,"text2terrain");
    SetTargetFPS(60);

    Camera cam = {0};
    init_camera(&cam);
    
    float minh = height[0], maxh = height[0];
    for(int i = 1; i < RES*RES; i++){
        if(height[i] < minh) minh = height[i];
        if(height[i] > maxh) maxh = height[i];
    }
    printf("Height range: %.2f to %.2f\n", minh, maxh);
    float rangeh = (maxh - minh) > 1e-6f ? (maxh - minh) : 1.0f;

    while(!WindowShouldClose()){
        BeginDrawing();
        ClearBackground(BLACK);
        first_person_control(&cam);
        BeginMode3D(cam);
        for(int y = 0; y < RES; y++){
            for(int x = 0; x < RES; x++){
                int k = y*RES + x;
                float h = height[k];
                
                // Normalize height to 0-1 range for better visualization
                float normalized_h = (h - minh) / rangeh;
                
                // Scale terrain for better visibility
                float terrain_scale = 1.0f;
                //float height_scale = rangeh > 1000.0f ? 0.001f : 0.01f;
                float height_scale = 1.f;
                
                Vector3 p = { 
                    ((float)x - RES/2.0f) * terrain_scale / RES, 
                    normalized_h * height_scale, 
                    ((float)y - RES/2.0f) * terrain_scale / RES 
                };
                
                // Color based on height - blue low, green mid, white high
                Color col;
                if (normalized_h < 0.3f) {
                    col = (Color){ 0, (unsigned char)(normalized_h * 255 / 0.3f), 255, 255 };
                } else if (normalized_h < 0.7f) {
                    float t = (normalized_h - 0.3f) / 0.4f;
                    col = (Color){ (unsigned char)(t * 139), (unsigned char)(255 - t * 116), (unsigned char)(255 - t * 255), 255 };
                } else {
                    float t = (normalized_h - 0.7f) / 0.3f;
                    col = (Color){ (unsigned char)(139 + t * 116), (unsigned char)(139 + t * 116), (unsigned char)(t * 255), 255 };
                }
                
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