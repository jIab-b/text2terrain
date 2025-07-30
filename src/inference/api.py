"""
FastAPI server for Text2Terrain inference.

Provides REST API endpoints for terrain generation from text descriptions.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import base64
import io
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .sampler import TerrainSampler


# Pydantic models for API
class TerrainRequest(BaseModel):
    text: str = Field(..., description="Natural language terrain description")
    world_x: int = Field(0, description="World X coordinate")
    world_y: int = Field(0, description="World Y coordinate")
    global_seed: int = Field(42, description="Global random seed")
    use_text_seed: bool = Field(True, description="Use deterministic seeds from text")
    return_image: bool = Field(False, description="Return base64-encoded PNG image")


class TerrainResponse(BaseModel):
    text: str
    heightmap_shape: List[int]
    heightmap_stats: Dict[str, float]
    module_ids: List[int]
    module_names: List[str]
    parameters: Dict[str, float]
    world_coordinates: List[int]
    generation_time: float
    heightmap_image: Optional[str] = None  # Base64-encoded PNG


class GridRequest(BaseModel):
    text: str = Field(..., description="Terrain description")
    grid_size: int = Field(3, ge=1, le=5, description="Grid size (1-5)")
    center_x: int = Field(0, description="Center tile X coordinate")
    center_y: int = Field(0, description="Center tile Y coordinate")
    global_seed: int = Field(42, description="Global seed")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of terrain descriptions")
    world_x: int = Field(0, description="Starting world X coordinate")
    world_y: int = Field(0, description="Starting world Y coordinate")
    global_seed: int = Field(42, description="Global seed")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    tile_size: int


# Global sampler instance
sampler: Optional[TerrainSampler] = None


def create_app(
    model_path: str,
    tokenizer_path: str = None,
    tile_size: int = 256,
    cors_origins: List[str] = None
) -> FastAPI:
    """Create FastAPI application."""
    
    app = FastAPI(
        title="Text2Terrain API",
        description="Generate procedural terrain from natural language descriptions",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    if cors_origins is None:
        cors_origins = ["*"]  # Allow all origins for development
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize sampler
    global sampler
    
    @app.on_event("startup")
    async def startup_event():
        global sampler
        print("Initializing Text2Terrain sampler...")
        sampler = TerrainSampler(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            tile_size=tile_size
        )
        print("✓ Sampler initialized successfully")
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        global sampler
        
        return HealthResponse(
            status="healthy" if sampler is not None else "loading",
            model_loaded=sampler is not None,
            device=sampler.predictor.device if sampler else "unknown",
            tile_size=sampler.tile_size if sampler else tile_size
        )
    
    @app.post("/generate", response_model=TerrainResponse)
    async def generate_terrain(request: TerrainRequest):
        """Generate terrain from text description."""
        global sampler
        
        if sampler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            start_time = time.time()
            
            # Generate terrain
            result = sampler.generate_terrain(
                text=request.text,
                world_x=request.world_x,
                world_y=request.world_y,
                global_seed=request.global_seed,
                use_text_seed=request.use_text_seed
            )
            
            generation_time = time.time() - start_time
            
            # Prepare response
            response = TerrainResponse(
                text=result["text"],
                heightmap_shape=list(result["heightmap"].shape),
                heightmap_stats=result["heightmap_stats"],
                module_ids=result["module_ids"],
                module_names=result["module_names"],
                parameters=result["parameters"],
                world_coordinates=result["world_coordinates"],
                generation_time=generation_time
            )
            
            # Add base64 image if requested
            if request.return_image:
                response.heightmap_image = _heightmap_to_base64(result["heightmap"])
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    @app.post("/generate/grid")
    async def generate_grid(request: GridRequest):
        """Generate a grid of terrain tiles."""
        global sampler
        
        if sampler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            start_time = time.time()
            
            result = sampler.generate_tile_grid(
                text=request.text,
                grid_size=request.grid_size,
                center_x=request.center_x,
                center_y=request.center_y,
                global_seed=request.global_seed
            )
            
            generation_time = time.time() - start_time
            
            # Convert heightmaps to base64 images
            grid_images = {}
            for (tile_x, tile_y), heightmap in result["heightmaps"].items():
                grid_images[f"{tile_x},{tile_y}"] = _heightmap_to_base64(heightmap)
            
            return {
                "text": result["text"],
                "grid_size": result["grid_size"],
                "center_coordinates": result["center_coordinates"],
                "tile_size": result["tile_size"],
                "module_ids": result["module_ids"],
                "module_names": result["module_names"],
                "parameters": result["parameters"],
                "grid_images": grid_images,
                "generation_time": generation_time
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Grid generation failed: {str(e)}")
    
    @app.post("/generate/batch")
    async def generate_batch(request: BatchRequest):
        """Generate terrain for multiple text descriptions."""
        global sampler
        
        if sampler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if len(request.texts) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 texts per batch")
        
        try:
            start_time = time.time()
            
            results = []
            for i, text in enumerate(request.texts):
                result = sampler.generate_terrain(
                    text=text,
                    world_x=request.world_x + i * sampler.tile_size,
                    world_y=request.world_y,
                    global_seed=request.global_seed + i,
                    use_text_seed=True
                )
                
                # Convert heightmap to base64
                result["heightmap_image"] = _heightmap_to_base64(result["heightmap"])
                # Remove large numpy array
                del result["heightmap"]
                
                results.append(result)
            
            generation_time = time.time() - start_time
            
            return {
                "results": results,
                "batch_size": len(request.texts),
                "total_generation_time": generation_time
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")
    
    @app.get("/examples")
    async def get_examples():
        """Get example terrain generations."""
        global sampler
        
        if sampler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Get example predictions (no terrain generation)
            examples = sampler.predictor.get_example_predictions()
            
            return {
                "examples": examples,
                "note": "Use /generate endpoint to create actual terrain heightmaps"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get examples: {str(e)}")
    
    @app.get("/parameters")
    async def get_parameters():
        """Get available terrain parameters and modules."""
        global sampler
        
        if sampler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Get module and parameter information
            modules = sampler.engine.registry.list_modules()
            all_params = sampler.engine.registry.get_all_parameters()
            
            return {
                "modules": [{"id": mid, "name": name} for mid, name in modules],
                "parameters": {
                    name: {"min": min_val, "max": max_val, "default": default}
                    for name, (min_val, max_val, default) in all_params.items()
                },
                "tile_size": sampler.tile_size
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get parameters: {str(e)}")
    
    return app


def _heightmap_to_base64(heightmap: np.ndarray) -> str:
    """Convert heightmap to base64-encoded PNG."""
    from PIL import Image
    
    # Normalize to [0, 255] for visualization
    normalized = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
    heightmap_8bit = (normalized * 255).astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(heightmap_8bit, mode='L')
    
    # Save to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode as base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


def main():
    """CLI entry point for API server."""
    
    parser = argparse.ArgumentParser(description="Text2Terrain API Server")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer (optional)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server")
    parser.add_argument("--tile-size", type=int, default=256, help="Terrain tile size")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        return
    
    print(f"Starting Text2Terrain API server...")
    print(f"Model: {args.model_path}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    
    # Create app
    app = create_app(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        tile_size=args.tile_size
    )
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()