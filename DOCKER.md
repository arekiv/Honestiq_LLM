# Docker Setup for AI Text Detector

This project has been dockerized for easy deployment.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop the container
docker-compose down
```

The application will be available at `http://localhost:7860`

### Using Docker directly

```bash
# Build the image
docker build -t ai-text-detector .

# Run the container
docker run -p 7860:7860 ai-text-detector

# Run in detached mode
docker run -d -p 7860:7860 --name ai-detector ai-text-detector
```

## GPU Support

If you have NVIDIA GPU and want to use CUDA:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `Dockerfile` to use GPU-enabled base image:
   ```dockerfile
   FROM pytorch/pytorch:latest-cuda11.8.0-cudnn8-runtime
   ```

3. Uncomment GPU settings in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

4. Run with GPU:
   ```bash
   docker-compose up --build
   ```

## Environment Variables

The application supports configuration via `.env` file. Docker Compose will automatically load the `.env` file from the project root.

### Setting up .env file

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration:
   ```env
   PLAGIARISM_API_KEY=your_api_key_here
   PLAGIARISM_API_URL=https://www.prepostseo.com/apis/checkPlag
   DEV_MODE=false
   DEBUG=false
   ```

3. The `.env` file is automatically loaded by docker-compose

### Alternative: Pass environment variables directly

You can also pass environment variables directly in `docker-compose.yml` or via command line:
```bash
docker-compose run -e PLAGIARISM_API_KEY=your_key ai-detector
```

## Notes

- The application uses port 7860 by default (Gradio's default port)
- Model files (`.bin` files) are included in the Docker image
- File uploads are processed within the container
- The container runs with `share=True` which creates a public Gradio link (you may want to modify `app.py` to use `share=False` for production)
- Environment variables from `.env` are automatically loaded by docker-compose

