# Distributed Reinforcement learning on GKE

Hands on tutorial of implementing Actor-Learner architecture (Ape-X architecture) on Google Kubernetes Engine


## 1. Implementing Ape-X for single node with ray

`code/` is simple implmentation of Ape-X for `CartPole-v1` env.

```
# Example
pip install -f requirements.txt
python main.py --num-actors=4
```

## 2. Settingup GCP

a. install Google Cloud SDK

b. Create GCP Project

c. Basic settings

`gcoud auth login`

`gcloud config project`

(e.g. us-central1)
`gcloud config set compute/region <regionName>`

(e.g. us-central1-a)
`gcloud config set compute/zone <zoneName>`

## 3. Build and register docker image to GCR

ベースイメージが異なる

Bulid docker image for actor

Bulid docker image for learner



