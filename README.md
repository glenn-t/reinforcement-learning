# reinforcement-learning
Based on Udemy course Reinforcement Learning with Python, linked [here](https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/?LSNPUBID=Jbc0N5ZkDzk&ranEAID=Jbc0N5ZkDzk&ranMID=39197&ranSiteID=Jbc0N5ZkDzk-._h1PEob2obmKoVouzF9iQ)

### Building and running the programme
For normal running:
```
docker-compose up
````

If doing development, force source code refresh using:
```
docker-compose up --build
````

The image uses volumes to save analysis artifacts. These are saved to the `output` folder.

For debug purposes, can try the following:

  1. Put breakpoints in code using pdb
  2. Build image: `docker build -t rl-02 .`
  3. Run container in interactive mode with volumes attached. `docker container run --interactive -v './output:/app/output' rl-02`

Or the following also works, and might be easier:

  1. Put breakpoints in code using pdb
  2. Build and run image: `docker-compose up --build -d`
  3. Find container name: `docker ps` or `docker container ls`
  4. Attach to container: `docker attach [CONTAINER_NAME]`
