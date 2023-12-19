<a id='1'></a>
# Towards Enhanced Autonomous Driving Experience

In this project, our goal is to create a smart vehicle designed to address common challenges faced by drivers, enhancing the overall driving experience by mitigating difficulties encountered on the road.

<a id='2'></a>
# Table of Contents: 
* [Description.](#1)
* [Table of Contents.](#2)
* [Our Approach.](#3)
* [Block Diagrams.](#4)
  * [Autopilot mode of operation.](#5)
  * [Driver mode of operation.](#6)
  * [Internal Block Diagrams.](#7)
* [Directory Hierarchy.](#8)


<a id='3'></a>
# Our Approach: 
Our approach is to design a smart vehicle that features: 

* **Auto Pilot System**: ​
  * Perceiving surrounding Environment system.​
  * Prediction motion and behavior of  road agents.    ​
  * Localizing itself.​
  * Planning road from source to destination.​

* **ADAS**:​
  * Steer-by-wire.​
  * Adaptive headlight.​
  * Infotainment system.​

* **Driver Emergency  alert**​
  * Dashboard Alert System.​
  * Sleep alert.

-------

<a id='4'></a>
# Block Diagrams: 
Our project has two modes of operation: 
<a id='5'></a>
### Autopilot mode of operation: 
<p align="center">
  <img src="https://github.com/MahmoudEl-Husseni/Towards-Enhanced-Autonomous-Driving-Experience/assets/67474135/bd1f0bda-139a-4935-a7c2-aec82ce592c4" alt="Autopilot mode" border="0">
</p>

-------
<a id='6'></a>
### Driver mode of operation

<p align="center">
  <img src="https://github.com/MahmoudEl-Husseni/Towards-Enhanced-Autonomous-Driving-Experience/assets/67474135/2dc5cb6a-4fb4-4cff-ba3c-fd87655d8161" alt="Driver mode" border="0">
</p>

------

<a id='7'></a>
### Internal block diagrams
#### Agent Motion Prediction Architecture
<p align="center">
  <img src="https://github.com/MahmoudEl-Husseni/Towards-Enhanced-Autonomous-Driving-Experience/assets/67474135/db3ef660-3f38-4f1b-b416-de2d5fdad91b" alt="Agent Motion Prediction Architecture" border="0">
</p>

#### Autopilot Algorithms integration
<p align="center">
  <img src="https://github.com/MahmoudEl-Husseni/Towards-Enhanced-Autonomous-Driving-Experience/assets/67474135/035d4f1f-36c7-4fc8-b8c4-7d043c0d45af" alt="Autopilot Algorithms integration" border="0">
</p>


<p align="center">
  <img src="https://github.com/MahmoudEl-Husseni/Towards-Enhanced-Autonomous-Driving-Experience/assets/67474135/8a9e6cd3-2e36-4cd1-b24d-9f22eb314ba3" alt="Steer by wire" border="0">
</p>



<p align="center">
  <img src="https://github.com/MahmoudEl-Husseni/Towards-Enhanced-Autonomous-Driving-Experience/assets/67474135/1f2f4a4b-d7ac-4dab-ae2c-a9f7f464dcaf" alt="Adaptive Headlights" border="0">
</p>



<p align="center">
  <img src="https://github.com/MahmoudEl-Husseni/Towards-Enhanced-Autonomous-Driving-Experience/assets/67474135/83ceb002-749a-416f-8162-0acb4d7c5e6c" alt="Instrument Cluster" border="0">
</p>



<p align="center">
  <img src="https://github.com/MahmoudEl-Husseni/Towards-Enhanced-Autonomous-Driving-Experience/assets/67474135/6f84026c-6ae2-45d1-96df-cb0d9a74b281" alt="Infotainment System" border="0">
</p>

---------
<a id='8'></a>
# Directory Hierarchy
<pre>
  Project Root
  ├── AMP #(Agent Motion Prediction)
  │   ├── 
  │   │   ├── Datasets Exploration
  |   |   |   ├── Notebooks #(Tutorials)
  |   |   |   ├── papers    #(Collected papers about AMP)
  │   │   ├── CNN Motion-Experiments
  |   |   ├── README.md
  ├── ADAS #(Advanced driver assistance systems)
  │   ├── README.md
  ├── AV #(Autonomous Vehicle)
  │   ├── README.md
  ├── Dashboard
  │   ├── AV-Cluster
</pre>
