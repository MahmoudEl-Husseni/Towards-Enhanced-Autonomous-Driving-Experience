<a id='1'></a>
# BETR Implementation for Trajectory Prediction

I have trained this architecture on [**Argoverse2** Dataset](https://www.argoverse.org/av2.html) and tested it using suitable metrics (Mean Displacement Error `MDE`, Final Displacement Error `FDE`, Miss Rate `MR`).

---
<a id='2'></a>
# Table of Contents: 
* BETR Custom Implementation.
  * [Description.](#1)
  * [Table of Contents.](#2)
  * [Customization.](#3)
  * [Installation.](#4)
  * [Usage.](#5)
    * [Preprocess Argoverse Data.](#6)
    * [Train Model.](#7)
  * [Train process.](#10)
* Results
  * [Qualitative results.](#8)
  * [Quantitative results.](#9)


<a id='3'></a>
# Customization: 

## __First__
I have extracted my vector representation from data such that:


### Agent Vectors: 
$$V_i = [\begin{matrix} x_s & y_s & x_e & y_e & ts_{avg} & Candidate Density & vx_{avg} & vy_{avg} & heading_{avg} & P_{id}\end{matrix}]$$

### Object Vectors: 
$$V_i = [\begin{matrix} x_s & y_s & x_e & y_e & ts_{avg} & D_{a_{avg}} & \theta_{a_{avg}} & vx_{avg} & vy_{avg} & heading_{avg} & objtype & P_{id} \end{matrix}]$$

### Lane Vectors: 
$$V_i = [\begin{matrix} x_s & y_s & z_s & x_e & y_e & z_e & I(Intersection)& dir_{avg} & type & line_{id} \end{matrix}]$$

## **Second**
I have replaced `GNN based Graph Local Encoders` with other vector encoder for each type of vector using `transformers and attention mechanism`, I have also used **Positional Encoding** based on time step of each vector to hold time information.

I also replaced normal `Global GNN layer` with Attentional GNN to better encode vectors of each type.

I Used a simple **Decoder** consists of `three MLP layers` to compare it with other model regardeless of decoder architecture.




![fig.1 Global Architecture](images/BETR.png)

<p align="center">
  <em>fig.1 Global Architecture</em>
</p>

| ![Transformer inside design](images/transformer.png)| ![Dense Layer inside design](images/decoder.png) |
| ----------- | ----------- |

<p align="center">
  <em>Transformer inside design</em> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em>Dense Layer inside design</em> 
</p>

----
<a id='4'></a>
# Installation:

Required Python Libraries: 
<pre>
    torch-geometric
    kornia
    colorama
    universal-pathlib
    plotly
</pre>

<pre>bash
> pip3 install -r requirements.txt
</pre>


<a id='5'></a>
# Usage:

<a id='6'></a>
## Preprocess Argoverse Data: 
Build **Docker** image to preprocess data in multithreads

<pre>bash
> docker build -t preprocess .
</pre>

Run script to run processing image: 
<pre>run.sh
  #!/bin/bash
  echo passwd | sudo -S rm /usr/bin/logs
  cd /usr/bin
  echo passwd | sudo -S touch /usr/bin/logs
  echo passwd | sudo -S chmod +777 /usr/bin/logs
  echo "Starting..." | cat >> logs
  while ! ping -c 1 google.com; do
  	sleep 5
  done
  
  echo "Internet Available..." | cat >> logs
  echo passwd | sudo -S systemctl start docker 
  sleep 10s
  echo passwd | sudo -S docker ps -a >> logs

  docker run --rm -it -v "/home/mahmoud":/main --gpus all preprocess
</pre>

<pre>bash
> sh run.sh
</pre>


<a id='7'></a>
## Train model: 
Build **Docker** image to train model using processed data.
Edit in `config.py` file to be compatible with your own configurations

<pre>bash
> docker build -t train .
</pre>

Run script to run training image: 
<pre>run.sh
  #!/bin/bash
  echo passwd | sudo -S rm /usr/bin/logs
  cd /usr/bin
  echo passwd | sudo -S touch /usr/bin/logs
  echo passwd | sudo -S chmod +777 /usr/bin/logs
  echo "Starting..." | cat >> logs
  while ! ping -c 1 google.com; do
  	sleep 5
  done
  
  echo "Internet Available..." | cat >> logs
  echo passwd | sudo -S systemctl start docker 
  sleep 10s
  echo passwd | sudo -S docker ps -a >> logs

  docker run --rm -it -v "/home/mahmoud":/main --gpus all train
</pre>

<pre>bash
> sh run.sh
</pre>

---
<a id='10'></a>
## Train process: 
![alt text](images/train-loss.png)
<p align="center">
  <em>train loss</em>
</p>

![alt text](images/train-mde.png) 
<p align="center">
  <em>train mde</em>
</p>

![alt text](images/train-nll.png) 
<p align="center">
  <em>train nll</em>
</p>

![alt text](images/val-loss.png) 
<p align="center">
  <em>val loss</em>
</p>


![alt text](images/val-mde.png) 
<p align="center">
  <em>val mde</em>
</p>

![alt text](images/val-nll.png)
<p align="center">
  <em>val nll</em>
</p>


---
# Results

<a id='8'></a>
## Qualitative results:
|![alt text](images/success_ex2_pred.png)|![alt text](images/success_ex3_pred.png)|![alt text](images/success_ex_pred.png)|
|-|-|-|





<a id='9'></a>
## Quantitative results:

|![alt text](images/fde.png)|![alt text](images/ade.png)
|-|-|

<p align="center">
  <em>minimum (Average, Final) Displacement Error Comparison</em>
</p>

---

![alt text](images/model-size.png)
<p align="center">
  <em>Model Size comparison</em>
</p>


---
![alt text](images/inference-time.png)
<p align="center">
  <em>inference time comparison</em>
</p>
