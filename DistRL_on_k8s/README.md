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

```
#: Create CPU node-pool (8 vCPU 12 Gib RAM, autoscaling)
gcloud container clusters create rl-cluster --pre-emptible --autoscale

#: Create GPU node-pool (1 node only)
gcloud container node-pool

#: Install GPU driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## 検証

1nodeで6CPU
rayの動作検証：cliでray start, ray init(adree="auto")でOKっぽい

---
1. 大きめのプリエンティブルクラスタ作成
2. debug-podからnginxを名前解決できるか確認(curl http://master-svc)
3. 1node-1podで6CPU PVC(delete)
4. Nnode-Npodで6CPU PVC(delete) → CPUのみ小規模テスト実行

5. GPU node → GPUあり小規模テスト実行
6. CPUノードプールのオートスケール 本番実行

redispassはデフォルトで同じ
ray start --address='10.8.0.11:6379' --redis-password='5241590000000000'

## 5. Monitoring

`kubectl get svc master-svc`

tensorboard: `EXTERNAL-IP:6006`

ray-dashboard: `EXTERNAL-IP:8265`

## 6. Delete cluster

`gcloud container clusters delete rl-cluster`

## References

https://cloud.google.com/tpu/docs/tutorials/kubernetes-engine-resnet

https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-provisioning#gcloud

https://docs.ray.io/en/latest/cluster/cloud.html#starting-ray-on-each-machine
