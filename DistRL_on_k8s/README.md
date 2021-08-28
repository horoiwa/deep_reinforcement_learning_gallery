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

<br>

## 4. Launch GKE cluster

https://cloud.google.com/sdk/gcloud/reference/container/clusters/create

```
#: Create GPU node-pool (1 node only)
gcloud container clusters create rl-cluster \
    --accelerator type=nvidia-tesla-p4, count=1 \
    --preemptible --num-nodes 1 \
    --machine-type "custom-16-32768"

#: Create CPU node-pools for actor processes.
gcloud container clusters node-pools create cpu-node-pool \
    --cluster rl-cluster \
    --preemptible --num-nodes 3 \
    --machine-type "custom-16-32768" \
    --enable-autoscaling --min-nodes 0 --max-nodes 30 \


gcloud container clusters get-credentials rl-cluster

#: Install GPU driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

Login to master container & start learning

`kubectl apply -f apex-cluster.yml`

`kubectl exec -it master bash`

`python /code/main.py --logdir log/tfboard --cluster --num_actors 100 --num_iters 30000`

<br>

## 5. Monitoring

`kubectl get svc master-svc`

tensorboard: `EXTERNAL-IP:6006`

ray-dashboard: `EXTERNAL-IP:8265`

<br>

## 6. Delete cluster

`gcloud container clusters delete rl-cluster`

<br>
