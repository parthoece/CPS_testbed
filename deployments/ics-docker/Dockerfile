FROM ubuntu:20.04

# Create src directory
RUN mkdir /src

# Copy source files
COPY ./src/ /src/

# Update and install required packages
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
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
    rsyslog && \
    pip install pyModbusTCP
    
# Add /usr/sbin to PATH to ensure rsyslogd is found
ENV PATH="/usr/sbin:$PATH"
# Copy the custom rsyslog configuration
#COPY ./rsyslog.conf /etc/rsyslog.conf

# Set working directory
WORKDIR /src

# Start rsyslog in foreground
CMD ["rsyslogd", "-n"]


