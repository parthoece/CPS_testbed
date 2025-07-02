#!/bin/bash
for container in plc1 plc2 hmi1 hmi2 hmi3 attacker attackermachine attackerremote; do
    docker exec $container logger -n 192.168.0.10 -P 514 "Test log from $container"
done

