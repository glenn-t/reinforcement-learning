# reinforcement-learning
Based on Udemy course Reinforcement Learning with Python, linked [here](https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/?LSNPUBID=Jbc0N5ZkDzk&ranEAID=Jbc0N5ZkDzk&ranMID=39197&ranSiteID=Jbc0N5ZkDzk-._h1PEob2obmKoVouzF9iQ)

### Building the image
```
docker build . -t reinforcement-learning
```

### Running the image
```
docker container run --publish 3000:3000 --rm --attach STDOUT reinforcement-learning
```
