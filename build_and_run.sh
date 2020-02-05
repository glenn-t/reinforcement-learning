docker build . -t reinforcement-learning
docker container run --publish 3000:3000 --rm --interactive reinforcement-learning