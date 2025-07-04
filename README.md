

A cyber-physical systems (CPS) testbed that emulates a water-filling factory process.  
> The architecture includes Human-Machine Interfaces (HMIs), PLC logic, and attacker nodes 

---

##  Table of Contents

1. [System Emulation Architecture](#system-emulation-architecture)
2. [Testbed Capabilities](#testbed-capabilities)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Running the Testbed](#running-the-testbed)
6. [Interacting with Containers](#interacting-with-containers)
7. [Starting Attacks](#starting-attacks)
8. [Docker Cheatsheet](#docker-cheatsheet)
9. [Optional: Docker Files](#optional-docker-files)
10. [References](#references)
11. [License](#license)

---

##  System Emulation Architecture

### Overall System Setup

![System Architecture](/assets/testbed.png)  
*Figure 1 – CPS testbed architecture showing physical process simulation.*

---

### System Fault Generation Flow

![System Fault Flow](/assets/fault.png)  
*Figure 2 – Workflow to generate regular system faults during operation, simulating sensor or actuator-level issues.*

---

### Cyber Attack Generation Flow

![Cyber Attack Flow](/assets/attack.png)  
*Figure 3 – Cyber attack sequence showing interaction between PLC, HMIs, attackers, and physical process simulation.*



##  Testbed Capabilities

This testbed can simulate and generate **three distinct types of operational data**:

1. **Normal Operation**
2. **Regular System Faults Operation**
3. **Cyber Attack Scenario**

>  **Important**:  
> Before starting **each new operation**, ensure the following:
> - Move or archive previous **PLC snapshot**, **syslog**, and **network traffic capture files**.  
> - Start each mode **independently** to isolate data for analysis.  
> - For **Cyber Attack**, begin with a **normal operation startup**, then activate the **attacker machine** to deploy all attacks in sequence.

---

##  Prerequisites

| Tool            | Version tested | Notes                          |
|-----------------|----------------|--------------------------------|
| Ubuntu / Kali   | 20.04 LTS / 2024 | WSL 2 works too               |
| Git             | ≥ 2.34         | Install via PPA                |
| Docker Engine   | ≥ 25.0         | Required for container support |
| Docker Compose  | v2             | Bundled with Docker            |
| Wireshark       | ≥ 4.2 (opt.)   | For packet capture             |

---


##  Installation
### clone the testbed
```
git clone https://github.com/parthoece/CPS_testbed.git
```

### 1 — Install Git
```bash
sudo add-apt-repository ppa:git-core/ppa
sudo apt-get update
sudo apt-get -y install git
```

### 2 — Install Docker

*Ubuntu 20.04*:  
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04

*Kali Linux*:  
https://www.kali.org/docs/containers/installing-docker-on-kali/

```bash
sudo apt clean && sudo apt autoremove
docker version
```

### 3 — Install Wireshark (optional)
```bash
sudo apt update && sudo apt upgrade
sudo apt install wireshark
sudo usermod -aG wireshark $USER
su - $USER
export DISPLAY=:0
xclock
```

### 4 — Make Shell Scripts Executable
```bash
find ./CPS_testbed -type f -name "*.sh" -exec chmod 555 {} \;
# or (to retain write permission)
find ./CPS_testbed -type f -name "*.sh" -exec chmod +x {} \;
```

---

##  Running the Testbed

```bash
# Confirm docker group membership
groups

# Verify current directory
pwd

# WSL checks
wsl.exe -l -v
docker context ls

# Login to Docker Hub if needed
docker login

# Check Docker service
sudo systemctl status docker
```

### Start / Stop the Testbed
```bash
cd CPS_testbed/deployments
sudo ./init_normal.sh              # start the simulation
docker compose down              # stop and remove containers
docker compose up --build        # build and start from scratch
sudo docker commpose ps
```

---

##  Interacting with Containers

### Open a Shell Inside Any Container
```bash
docker exec -it <container-name> bash
```

### Run HMI Applications
To avoid generating the same data, engage all 3 HMI during each operational mode. 

1. **HMI1:** real-time physical process status( Actuators, Sensors).
2. **HMI2:** manual instruction to mimic human in the loop( actuators, sensors).
3. **HMI3:** auto mode (Specific ranges fixed  for Min-max sensor level)

   ![HMI Interaction during Normal & Fault mode](/assets/hmi.png)  
*Figure 4 – HMI User Interaction - HMI1, HMI2, HMI3.*
> Under normal operation, the sensor status changes according to the command sent.
> Also verify system faults – by sending control command vs real-time response
> Here, sensor drift and valve sticking faults are visible

```bash
sudo docker exec -it hmi2 python3 /src/HMI1.py
sudo docker exec -it hmi2 python3 /src/HMI2.py
sudo docker exec -it hmi2 python3 /src/HMI3.py
```

---

##  Starting Attack Modules (Kali Linux)

Start cyber attack simulation by engaging these modules (after normal start):
```bash
sudo docker exec -it attackermachine python3 /src/AttackerMachine.py
sudo docker exec -it attacker        python3 /src/Attacker.py
```
### To attack remotely, configure the MQTT connection
```
sudo docker exec -it attackerremote  python3 /src/AttackerRemote.py
```

---

##  Docker Cheatsheet

| Action                     | Command |
|----------------------------|---------|
| Run container              | `docker run -it/-d/-t …` |
| Start / stop container     | `docker start/stop <name>` |
| List containers/images     | `docker ps -a`, `docker images` |
| Remove unused resources    | `docker system prune -a` |
| Build image manually       | `docker build -t <tag> .` |
| Compose up (detached)      | `docker compose -f docker-compose.yml up -d` |

---

##  Optional: Docker Files

- `Dockerfile` — for building custom container images.
- `docker-compose.yml` — for defining and orchestrating multi-container simulation environments.

---

##  References

1. DigitalOcean. *How to Install and Use Docker on Ubuntu 20.04*  
2. Kali Linux. *Installing Docker on Kali Linux*  
3. Git-SCM. *Pro Git Book*  
4. Wireshark Foundation. *Wireshark User’s Guide*  
5. NIST. *Guide to Industrial Control Systems (ICS) Security*, SP 800-82 Rev 2  
6. Dehlaghi-Ghadim, A. et al. (2023). *CPS_testbed — A framework for building industrial control systems security testbeds*, Computers in Industry, 148, 103906. https://doi.org/10.1016/j.compind.2023.103906

---

##  License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for full text.

---

> Maintained by **Partho Adhikari / High Speed Network Lab**. Contributions welcome!

