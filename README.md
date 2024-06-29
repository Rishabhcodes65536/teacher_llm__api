# ExamBuddy API

This repository contains the API for ExamBuddy, an AI-powered tutoring platform designed to help users with math problems. The API is dockerized and uses Ollama for model storage and GPU acceleration.

## Getting Started

### Prerequisites

- Docker
- Web Services for hosting server Azure for Our case

### Setup

0. **Initializations**
Best practice
```bash
sudo apt update
sudo apt upgrade
```

1. **Clone the Repository:**
    Setup your server Ubuntu for our case
```bash
git clone https://github.com/Rishabhcodes65536/teacher_llm__api.git
cd test_app
```

2. **Build the Docker Image:**
```bash 
 sudo docker-compose up --build -d  api_gateway
 ```

3. **Ollama setup**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama pull llama3
```

Note:
In our case Ollama is initialised to setup on port 11434 if processes are running,YOU NEED TO KILL THEM!!
```bash
lsof -i :11434 -t
kill -9 <pid>
```

4. **Test it DIY!!**
No internet is required XD.Flaunt your Prompting skills to mine creativity!!

``` bash
ollama serve
ollama run llama3
```

Prompt it! Conquer it!

```bash
/bye
``` 

To stop the prompt interface

5. **Check your computational Power GPU is highly recommended**
```bash
watch -n 0.5 nvidia-smi
```
We have R&D on CPU v/s GPU performance & we found approximately 900% improvement in performance
On running prompt almost 87-90% consumption of GPU power was observed.

6. **Creating ollama container**
```bash
docker-compose up -d --build ollama
```

7. **Checking & running the container**
```bash
 docker ps
 docker run -d -v ollama:/root/.ollama -p 8084:8084 --name ollama ollama/ollama
 ```

 8. **Starting the api services**
 ```bash
  sudo docker system prune
  sudo docker-compose up --build -d  api_gateway
  ```
  First we manage cached docker containers and images by pruning it before starting with build
  Note: Building first time may require some time,but subsequent builds wont require much time.

9. **Testing the API services**
Example testing
```bash
curl -X POST "YOUR_SERVER_IP:YOUR_PORT/internal/question_generation/analyse/topic" \
  -H "Accept: */*" \
  -H "Content-Type: application/json" \
  -d '{"question":Integration}'
```

10. **Stopping our services**
``` bash
sudo docker system prune
sudo docker-compose api_gateway
```

11. **View logs**
```bash
sudo docker-compose logs -f --tail 1000 api_gateway
```








