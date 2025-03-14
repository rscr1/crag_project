#!/bin/bash 
sudo docker run --rm -id --shm-size 64G --gpus '"device=0, 1, 2, 3"' -p 11434:11434 --name ollama_cont ollama:v0 
