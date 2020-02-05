# reinforcement-learning
Based on Udemy course Reinforcement Learning with Python

### Building the image
```
docker build . -t reinforcement-learning
```

### Running the image
```
docker container run --publish 3000:3000 --rm --attach STDOUT reinforcement-learning
```
