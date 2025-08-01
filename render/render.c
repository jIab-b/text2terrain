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
/*
static inline float clampf(float v, float lo, float hi){
    return v < lo ? lo : (v > hi ? hi : v);
}
*/
static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}


typedef struct OrbitCamera {
    Camera3D camera;
    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    bool is_panning;
    float rotation_sensitivity;
    float pan_sensitivity;
    Vector2 last_mouse_pos;
} OrbitCamera;

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



static void update_camera_position(OrbitCamera *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

        float offsetX = r * cosf(el) * cosf(az);
    float offsetY = r * cosf(el) * sinf(az);
    float offsetZ = r * sinf(el);

    Vector3 offset = {offsetX, offsetY, offsetZ};
    c->camera.position = Vector3Add(c->camera.target, offset);
}

static void player_movement(OrbitCamera *client){
    float speed = 0.1f * client->camera_distance;
    if(IsKeyDown(KEY_LEFT_SHIFT)) speed *= 10.0f;
    Vector3 move = {0};
    Vector3 forward = Vector3Subtract(client->camera.target, client->camera.position);
    forward.y = 0;
    if(Vector3LengthSqr(forward) < 0.000001f) forward = (Vector3){0,0,-1};
    forward = Vector3Normalize(forward);
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, (Vector3){0,1,0}));
    if(IsKeyDown(KEY_W)) move = Vector3Add(move, Vector3Scale(forward, speed));
    if(IsKeyDown(KEY_S)) move = Vector3Add(move, Vector3Scale(forward, -speed));
    if(IsKeyDown(KEY_D)) move = Vector3Add(move, Vector3Scale(right, speed));
    if(IsKeyDown(KEY_A)) move = Vector3Add(move, Vector3Scale(right, -speed));
    client->camera.target = Vector3Add(client->camera.target, move);
    if(Vector3LengthSqr(move) > 0.0f) update_camera_position(client);
}

void handle_camera_controls(OrbitCamera *client) {
    Vector2 mouse_pos = GetMousePosition();
    player_movement(client);

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};

        float sensitivity = client->rotation_sensitivity;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    // Right mouse button panning
    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
        client->is_panning = true;
        client->last_mouse_pos = mouse_pos;
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_RIGHT)) {
        client->is_panning = false;
    }

    if (client->is_panning && IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};
        float ps = client->pan_sensitivity * client->camera_distance;


   
        Vector3 forward = Vector3Normalize(Vector3Subtract(client->camera.target, client->camera.position));
        Vector3 worldUp = (Vector3){0, 1, 0};
        Vector3 right = Vector3CrossProduct(forward, worldUp);
        if (Vector3LengthSqr(right) < 0.000001f) right = Vector3CrossProduct(forward, (Vector3){1, 0, 0});
        right = Vector3Normalize(right);
        Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));

        Vector3 pan_offset = Vector3Add(
            Vector3Scale(right, -mouse_delta.x * ps),
            Vector3Scale(up, mouse_delta.y * ps)
        );
        client->camera.target = Vector3Add(client->camera.target, pan_offset);
   
        /*
        
        float forward_x = -cosf(client->camera_azimuth);
        float forward_y = -sinf(client->camera_azimuth);
        float right_x = -sinf(client->camera_azimuth);
        float right_y =  cosf(client->camera_azimuth);
        
        client->camera.target.x += (-mouse_delta.x * ps) * right_x + (mouse_delta.y * ps) * forward_x;
        client->camera.target.y += (-mouse_delta.x * ps) * right_y + (mouse_delta.y * ps) * forward_y;
 
        */
        client->last_mouse_pos = mouse_pos;
        
        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
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
    OrbitCamera cam={0};
        cam.camera_distance = 30.0f;
    cam.camera_azimuth = 0.0f;
    cam.camera_elevation = 0.3f;
        cam.is_dragging = false;
    cam.is_panning = false;
    cam.rotation_sensitivity = 0.000005f;
    cam.pan_sensitivity = 0.000005f;
    cam.camera.up = (Vector3){0,1,0};
    cam.camera.fovy = 45;
    cam.camera.projection = CAMERA_PERSPECTIVE;
    update_camera_position(&cam);
    DisableCursor();
    while(!WindowShouldClose()){
        handle_camera_controls(&cam);
        BeginDrawing();
        ClearBackground(BLACK);
        BeginMode3D(cam.camera);
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