# Use Ubuntu as the base image
FROM ubuntu:20.04

# Create a working directory inside the container
RUN mkdir /src

# Copy source files into the container
COPY ./src/ /src/

# Set environment variables to handle time zone setup non-interactively
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    tzdata \
    sudo \
    python3 \
    iputils-ping \
    net-tools \
    git \
    nano \
    python3-pip \
    telnet \
    memcached \
    python3-memcache \
    ettercap-common \
    nmap \
    python3-scapy \
    rsyslog && \
    pip install pyModbusTCP paho-mqtt

# Add /usr/sbin to PATH to ensure rsyslogd is found
ENV PATH="/usr/sbin:$PATH"

# Copy the custom rsyslog configuration
#COPY ./rsyslog.conf /etc/rsyslog.conf

# Set the working directory
WORKDIR /src

# Uncomment the following line if you need a custom startup script
# COPY ./start.sh ./start.sh

# Set the default command for the container
CMD ["rsyslogd", "-n"]



