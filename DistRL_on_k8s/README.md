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

https://cloud.google.com/sdk/gcloud/reference/container/clusters/create

```
#: Create CPU node-pool for actor processes.
gcloud container clusters create rl-cluster \
    --preemptible --num-nodes 3 \
    --machine-type "e2-standard-8" \
    --enable-autoscaling --min-nodes 3 --max-nodes 30 \

#: Create GPU node-pool (1 node only) for buffer and learner process.

gcloud container node-pool create gpu-pool \
    --accelerator nividia-tesla-p4  \

#: Install GPU driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## 検証

1nodeで6CPU
rayの動作検証：cliでray start, ray init(adree="auto")でOKっぽい

---
1. 大きめのプリエンティブルクラスタ作成 -> OK
2. debug-podからnginxを名前解決できるか確認
   curl http://master-svc
   -> OK, ただしnslookup master-svc はロードバランサのIPを返す
   -> ロードバランサとは別にheadless svcを立てるとpodのIPに一致し、curl http://(IP=nslookup ray-svc) で通る
   nslookup ray-headless-svc | grep Address | tail -n +2 | cut -f2 -d ' '

3. 1node-1podで6CPU PVC(delete)
4. Nnode-Npodで6CPU PVC(delete) → CPUのみ小規模テスト実行
    -> うまくCPUを認識しないので明示的にstart時に与える必要があった。
    ray.init()はコマンドラインでray startしてれば不要

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
