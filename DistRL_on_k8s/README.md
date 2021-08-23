# Distributed Reinforcement learning on GKE

Hands-on tutorial of Actor-Learner architecture (Ape-X architecture) with ray on Google Kubernetes Engine


## 1. Simple Ape-X implementation with ray

`code/` is simple implmentation of Ape-X for `CartPole-v1` env.

```
#: Run on local machine
pip install -f requirements.txt
python main.py --num-actors=4 --logdir=log
```

<br>

## 2. Setup

#### Install Cloud SDK, docker, kubectl on your local machine

Cloud SDK:
https://cloud.google.com/sdk/docs/install

docker:
https://docs.docker.com/get-docker/

kubectl:
https://kubernetes.io/ja/docs/tasks/tools/install-kubectl/

<br>

#### Create new GCP project

Create GCP account:<br>
https://cloud.google.com/apigee/docs/hybrid/v1.1/precog-gcpaccount

<br>

```
gcloud auth login

#: gcloud projects create <ProjectID> --name <ProjectName>
gcloud projects create distrl-project --name distrl
```

<br>

From GCP web console, add billing information to the project and enable `ComputeEngine API`, `ContainerRegistry API` and `Kubernetes Engine API`

<br>

#### Project config

```
#: gcloud config set project <ProjectID>
gcloud config set project distrl-project

#: gcloud config set compute/region <RegionName>
gcloud config set compute/region northamerica-northeast1

#: gcloud config set compute/zone <zoneName>
gcloud config set compute/zone northamerica-northeast1-a


gcloud config list
```

<br>

## 3. Build and register docker image to GCR

Bulid docker image and push to GCR

```
gcloud auth configure-docker

docker build -t gcr.io/distrl-project/distrl .

docker push gcr.io/distrl-project/distrl
```




## References

https://cloud.google.com/tpu/docs/tutorials/kubernetes-engine-resnet
