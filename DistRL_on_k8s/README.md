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


## 4. Launch GKE cluster

step1. 1nodeで6CPU
step2. 1nodeで6CPU PVC(delete)
step3. 2nodeで6CPU, 1GPU, PVC(delete)
step4. 1GPUnode + autoscale nodeで 2CPU/1GPU, 64CPU, PVC(delete)

クラスタオートスケール（ノード数の自動スケール）なのかノード自動プロビジョニング（ノードプール）なのかはわからん

```
gcloud container clusters create rl-cluster --pre-emptible --autoscale

kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

`ray start --head --dashboard-host "0.0.0.0"`

## Monitoring

`kubectl get svc master-svc`

tensorboard: `EXTERNAL-IP:6006`

ray-dashboard: `EXTERNAL-IP:8265`

## References

https://cloud.google.com/tpu/docs/tutorials/kubernetes-engine-resnet

https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-provisioning#gcloud

https://docs.ray.io/en/latest/cluster/cloud.html#starting-ray-on-each-machine
