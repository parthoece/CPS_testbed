#!/bin/bash

printStep(){
    echo ""
    echo "[" $1 "STARTED]"
    sleep 1
}

# Check and set the simulation mode
FORCE_PROMPT=${FORCE_PROMPT:-true}

setSimulationMode(){
    MODE_FILE="./mode.conf"

    if [[ "$FORCE_PROMPT" == "true" ]]; then
        echo "Please select the simulation mode."
        echo "1) Normal Operation"
        echo "2) System Faults"
        echo -n "Enter choice (1 or 2): "
        read choice

        if [[ $choice == "1" ]]; then
            SIMULATION_MODE="normal"
        elif [[ $choice == "2" ]]; then
            SIMULATION_MODE="faults"
        else
            echo "Invalid choice! Defaulting to 'normal' mode."
            SIMULATION_MODE="normal"
        fi

        echo "mode=$SIMULATION_MODE" > "$MODE_FILE"
        echo "Simulation mode set to: $SIMULATION_MODE"
    else
        if [[ -f "$MODE_FILE" && -s "$MODE_FILE" ]]; then
            SIMULATION_MODE=$(grep '^mode=' "$MODE_FILE" | cut -d '=' -f2)
        else
            SIMULATION_MODE="normal"
            echo "mode=$SIMULATION_MODE" > "$MODE_FILE"
        fi
    fi
}

# Update rsyslog.conf dynamically
updateRsyslogConfig(){
    
    RSYSLOG_CONF="./rsyslog/rsyslog.conf"
    LOG_DIR="/var/log/$SIMULATION_MODE/"

    mkdir -p "$LOG_DIR"
    chmod -R 755 "$LOG_DIR"

    # # Dynamically update only the relevant lines in rsyslog.conf
    sudo sed -i "/^\\\$template ContainerLogs/c\\\$template ContainerLogs,\"$LOG_DIR/%syslogtag%.log\"" $RSYSLOG_CONF
    sudo sed -i "/^\\\$template AggregatedLogs/c\\\$template AggregatedLogs,\"$LOG_DIR/all_logs.log\"" $RSYSLOG_CONF
    # Ensure logging rules are present and correct
    if ! grep -q ".* ?ContainerLogs" $RSYSLOG_CONF; then
        echo "*.* ?ContainerLogs" | sudo tee -a $RSYSLOG_CONF > /dev/null
    fi

    if ! grep -q ".* ?AggregatedLogs" $RSYSLOG_CONF; then
        echo "*.* ?AggregatedLogs" | sudo tee -a $RSYSLOG_CONF > /dev/null
    fi
}


# Set simulation mode
setSimulationMode
updateRsyslogConfig

printStep "DEPLOYMENT"

printStep "DOWN PREVIOUS CONTAINERS"
sudo docker compose down

printStep "CREATE TEMP SRC FILE"
sudo mkdir -p ./ics-docker/src/
sudo mkdir -p ./attacker-docker/src/

printStep "PRUNING DOCKER"
sudo docker system prune -f

printStep 'DOCKER_COMPOSE BUILD'
sudo docker compose build

printStep "REMOVE TEMP SRC FILE"
sudo rm -r ./ics-docker/src/
sudo rm -r ./attacker-docker/src/

printStep 'DOCKER_COMPOSE UP'
sudo docker compose up -d

printStep 'DOCKER_COMPOSE UP STATUS'
sudo docker compose ps

# Start capturing traffic packets based on mode
if [[ $SIMULATION_MODE == "normal" ]]; then
    printStep "CAPTURING NORMAL TRAFFIC PACKETS"
    sudo tcpdump -w ./logs/normal_traffic_$(date +%F_%T).pcap -i br_icsnet &
elif [[ $SIMULATION_MODE == "faults" ]]; then
    printStep "CAPTURING FAULTS TRAFFIC PACKETS"
    sudo tcpdump -w ./logs/faults_traffic_$(date +%F_%T).pcap -i br_icsnet &
fi

# Monitor simulation based on mode
if [[ $SIMULATION_MODE == "faults" ]]; then
    printStep "MONITORING FAULTS SIMULATION"
    sleep 3600
else
    printStep "MONITORING NORMAL OPERATION"
    sleep 3600
fi

printStep "DEPLOYMENT COMPLETED"
