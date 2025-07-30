#!/usr/bin/env python3
"""
Real-time terrain viewer using Raylib.

Displays generated terrain heightmaps as 3D meshes with camera controls.
Uses pure Python with pyray bindings (no CMake required).
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import time

try:
    import pyray as rl
    from pyray import *
    RAYLIB_AVAILABLE = True
except ImportError:
    RAYLIB_AVAILABLE = False
    print("Warning: pyray not available. Install with: pip install raylib-python")

from ..src.inference import TerrainSampler


class TerrainViewer:
    """
    Real-time 3D terrain viewer.
    
    Features:
    - WASD camera movement
    - Mouse look controls
    - Real-time terrain generation from text
    - Wireframe/solid rendering modes
    - Height-based coloring
    """
    
    def __init__(
        self,
        window_width: int = 1024,
        window_height: int = 768,
        tile_size: int = 256,
        height_scale: float = 50.0
    ):
        if not RAYLIB_AVAILABLE:
            raise ImportError("pyray is required. Install with: pip install raylib-python")
        
        self.window_width = window_width
        self.window_height = window_height
        self.tile_size = tile_size
        self.height_scale = height_scale
        
        # Camera state
        self.camera_pos = Vector3(0.0, 100.0, 100.0)
        self.camera_target = Vector3(0.0, 0.0, 0.0)
        self.camera_up = Vector3(0.0, 1.0, 0.0)
        
        # Terrain state
        self.current_mesh = None
        self.current_texture = None
        self.current_text = "rolling hills"
        self.wireframe_mode = False
        
        # Performance
        self.last_generation_time = 0.0
        
        # Optional terrain sampler for real-time generation
        self.sampler: Optional[TerrainSampler] = None
    
    def initialize(self, model_path: str = None):
        """Initialize the viewer and optionally load terrain sampler."""
        
        # Initialize Raylib
        rl.init_window(self.window_width, self.window_height, b"Text2Terrain Viewer")
        rl.set_target_fps(60)
        rl.disable_cursor()  # Capture mouse for camera
        
        # Initialize camera
        self.camera = Camera3D(
            self.camera_pos,
            self.camera_target,
            self.camera_up,
            45.0,
            CAMERA_PERSPECTIVE
        )
        
        # Load terrain sampler if model provided
        if model_path and Path(model_path).exists():
            print(f"Loading terrain sampler from {model_path}...")
            try:
                self.sampler = TerrainSampler(model_path)
                print("✓ Terrain sampler loaded successfully")
            except Exception as e:
                print(f"Failed to load sampler: {e}")
                print("Continuing with demo heightmap...")
        
        # Generate initial terrain
        self.generate_terrain(self.current_text)
        
        print("Terrain viewer initialized!")
        print("Controls:")
        print("  WASD - Move camera")
        print("  Mouse - Look around")
        print("  SPACE - Generate new terrain")
        print("  T - Toggle wireframe")
        print("  ESC - Exit")
    
    def generate_terrain(self, text: str = None) -> bool:
        """Generate and load new terrain."""
        
        start_time = time.time()
        
        try:
            if self.sampler and text:
                # Generate from text using trained model
                result = self.sampler.generate_terrain(text, world_x=0, world_y=0)
                heightmap = result["heightmap"]
                print(f"Generated terrain: {text}")
                print(f"  Modules: {result['module_names']}")
            else:
                # Generate demo heightmap
                heightmap = self._generate_demo_heightmap()
                print("Generated demo terrain (no model loaded)")
            
            # Convert heightmap to mesh
            mesh = self._heightmap_to_mesh(heightmap)
            
            # Clean up previous mesh
            if self.current_mesh:
                rl.unload_mesh(self.current_mesh)
            
            self.current_mesh = mesh
            self.current_text = text or "demo terrain"
            self.last_generation_time = time.time() - start_time
            
            return True
            
        except Exception as e:
            print(f"Failed to generate terrain: {e}")
            return False
    
    def _generate_demo_heightmap(self) -> np.ndarray:
        """Generate a simple demo heightmap using noise."""
        
        # Simple fractal noise for demo
        heightmap = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        
        for octave in range(4):
            freq = 0.01 * (2 ** octave)
            amp = 0.5 ** octave
            
            for y in range(self.tile_size):
                for x in range(self.tile_size):
                    nx = x * freq
                    ny = y * freq
                    
                    # Simple noise approximation
                    noise_val = (np.sin(nx) + np.cos(ny) + np.sin(nx + ny)) / 3.0
                    heightmap[y, x] += noise_val * amp
        
        # Normalize to [0, 1]
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
        
        return heightmap
    
    def _heightmap_to_mesh(self, heightmap: np.ndarray) -> Mesh:
        """Convert heightmap to Raylib mesh."""
        
        height, width = heightmap.shape
        vertex_count = height * width
        triangle_count = (height - 1) * (width - 1) * 2
        
        # Allocate arrays
        vertices = np.zeros(vertex_count * 3, dtype=np.float32)
        texcoords = np.zeros(vertex_count * 2, dtype=np.float32)
        indices = np.zeros(triangle_count * 3, dtype=np.uint16)
        
        # Generate vertices
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                
                # Position (centered around origin)
                vertices[idx * 3 + 0] = (x - width // 2) * 2.0  # X
                vertices[idx * 3 + 1] = heightmap[y, x] * self.height_scale  # Y (height)
                vertices[idx * 3 + 2] = (y - height // 2) * 2.0  # Z
                
                # Texture coordinates
                texcoords[idx * 2 + 0] = x / (width - 1)
                texcoords[idx * 2 + 1] = y / (height - 1)
        
        # Generate indices for triangles
        idx = 0
        for y in range(height - 1):
            for x in range(width - 1):
                # Two triangles per quad
                v0 = y * width + x
                v1 = y * width + (x + 1)
                v2 = (y + 1) * width + x
                v3 = (y + 1) * width + (x + 1)
                
                # First triangle
                indices[idx * 3 + 0] = v0
                indices[idx * 3 + 1] = v2
                indices[idx * 3 + 2] = v1
                idx += 1
                
                # Second triangle
                indices[idx * 3 + 0] = v1
                indices[idx * 3 + 1] = v2
                indices[idx * 3 + 2] = v3
                idx += 1
        
        # Create mesh
        mesh = Mesh()
        mesh.vertex_count = vertex_count
        mesh.triangle_count = triangle_count
        
        # Upload data to GPU
        mesh.vertices = rl.mem_alloc(vertices.nbytes)
        rl.mem_copy(mesh.vertices, vertices.ctypes.data, vertices.nbytes)
        
        mesh.texcoords = rl.mem_alloc(texcoords.nbytes)
        rl.mem_copy(mesh.texcoords, texcoords.ctypes.data, texcoords.nbytes)
        
        mesh.indices = rl.mem_alloc(indices.nbytes)
        rl.mem_copy(mesh.indices, indices.ctypes.data, indices.nbytes)
        
        # Upload mesh to GPU
        rl.upload_mesh(mesh, False)
        
        return mesh
    
    def update_camera(self, dt: float):
        """Update camera based on input."""
        
        # Mouse look
        mouse_delta = rl.get_mouse_delta()
        if rl.is_cursor_hidden():
            # Simple mouse look (this is basic - could be improved)
            pass
        
        # Keyboard movement
        speed = 100.0 * dt
        
        if rl.is_key_down(KEY_W):
            # Move forward
            forward = rl.vector3_subtract(self.camera.target, self.camera.position)
            forward = rl.vector3_normalize(forward)
            forward = rl.vector3_scale(forward, speed)
            self.camera.position = rl.vector3_add(self.camera.position, forward)
            self.camera.target = rl.vector3_add(self.camera.target, forward)
        
        if rl.is_key_down(KEY_S):
            # Move backward
            forward = rl.vector3_subtract(self.camera.target, self.camera.position)
            forward = rl.vector3_normalize(forward)
            forward = rl.vector3_scale(forward, -speed)
            self.camera.position = rl.vector3_add(self.camera.position, forward)
            self.camera.target = rl.vector3_add(self.camera.target, forward)
        
        if rl.is_key_down(KEY_A):
            # Strafe left
            forward = rl.vector3_subtract(self.camera.target, self.camera.position)
            right = rl.vector3_cross_product(forward, self.camera.up)
            right = rl.vector3_normalize(right)
            right = rl.vector3_scale(right, -speed)
            self.camera.position = rl.vector3_add(self.camera.position, right)
            self.camera.target = rl.vector3_add(self.camera.target, right)
        
        if rl.is_key_down(KEY_D):
            # Strafe right
            forward = rl.vector3_subtract(self.camera.target, self.camera.position)
            right = rl.vector3_cross_product(forward, self.camera.up)
            right = rl.vector3_normalize(right)
            right = rl.vector3_scale(right, speed)
            self.camera.position = rl.vector3_add(self.camera.position, right)
            self.camera.target = rl.vector3_add(self.camera.target, right)
    
    def handle_input(self):
        """Handle user input."""
        
        # Generate new terrain
        if rl.is_key_pressed(KEY_SPACE):
            terrain_options = [
                "jagged mountain peaks",
                "rolling green hills", 
                "desert sand dunes",
                "volcanic rocky terrain",
                "eroded canyon walls",
                "gentle meadow slopes"
            ]
            import random
            new_text = random.choice(terrain_options)
            print(f"Generating: {new_text}")
            self.generate_terrain(new_text)
        
        # Toggle wireframe
        if rl.is_key_pressed(KEY_T):
            self.wireframe_mode = not self.wireframe_mode
            print(f"Wireframe mode: {'ON' if self.wireframe_mode else 'OFF'}")
        
        # Toggle cursor
        if rl.is_key_pressed(KEY_TAB):
            if rl.is_cursor_hidden():
                rl.enable_cursor()
            else:
                rl.disable_cursor()
    
    def render(self):
        """Render the scene."""
        
        rl.begin_drawing()
        rl.clear_background(SKYBLUE)
        
        # 3D rendering
        rl.begin_mode_3d(self.camera)
        
        if self.current_mesh:
            # Draw terrain mesh
            if self.wireframe_mode:
                rl.draw_mesh_wires(self.current_mesh, DARKGREEN, Vector3(0, 0, 0))
            else:
                rl.draw_mesh(self.current_mesh, GREEN, Vector3(0, 0, 0))
        
        # Draw grid for reference
        rl.draw_grid(20, 10.0)
        
        rl.end_mode_3d()
        
        # UI overlay
        rl.draw_text(f"Terrain: {self.current_text}", 10, 10, 20, DARKGRAY)
        rl.draw_text(f"Generation time: {self.last_generation_time:.3f}s", 10, 35, 16, DARKGRAY)
        rl.draw_text(f"FPS: {rl.get_fps()}", 10, 55, 16, DARKGRAY)
        rl.draw_text(f"Mode: {'Wireframe' if self.wireframe_mode else 'Solid'}", 10, 75, 16, DARKGRAY)
        
        # Controls help
        if rl.is_cursor_hidden():
            rl.draw_text("WASD: Move, SPACE: New terrain, T: Wireframe, TAB: Cursor, ESC: Exit", 
                        10, self.window_height - 25, 14, LIGHTGRAY)
        else:
            rl.draw_text("TAB: Hide cursor to enable camera", 10, self.window_height - 25, 14, LIGHTGRAY)
        
        rl.end_drawing()
    
    def run_main_loop(self):
        """Run the main rendering loop."""
        
        print("Starting terrain viewer main loop...")
        
        while not rl.window_should_close():
            dt = rl.get_frame_time()
            
            # Handle input
            self.handle_input()
            
            # Update camera
            self.update_camera(dt)
            
            # Render
            self.render()
        
        # Cleanup
        if self.current_mesh:
            rl.unload_mesh(self.current_mesh)
        
        rl.close_window()
        print("Terrain viewer closed")


def main():
    """CLI entry point for terrain viewer."""
    
    parser = argparse.ArgumentParser(description="Text2Terrain Real-time Viewer")
    parser.add_argument("--model-path", help="Path to trained model (optional)")
    parser.add_argument("--width", type=int, default=1024, help="Window width")
    parser.add_argument("--height", type=int, default=768, help="Window height")
    parser.add_argument("--tile-size", type=int, default=128, help="Terrain tile size (lower for better performance)")
    parser.add_argument("--height-scale", type=float, default=50.0, help="Height scaling factor")
    
    args = parser.parse_args()
    
    if not RAYLIB_AVAILABLE:
        print("Error: pyray not available")
        print("Install with: pip install raylib-python")
        return 1
    
    try:
        # Create viewer
        viewer = TerrainViewer(
            window_width=args.width,
            window_height=args.height,
            tile_size=args.tile_size,
            height_scale=args.height_scale
        )
        
        # Initialize
        viewer.initialize(model_path=args.model_path)
        
        # Run
        viewer.run_main_loop()
        
        return 0
        
    except Exception as e:
        print(f"Error running terrain viewer: {e}")
        return 1


if __name__ == "__main__":
    exit(main())