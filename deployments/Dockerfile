FROM ubuntu:20.04

# Install rsyslog and dependencies
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    rsyslog tzdata && \
    rsyslogd -N1  # Validate rsyslog installation

# Add /usr/sbin to PATH (ensures rsyslogd is accessible)
ENV PATH="/usr/sbin:$PATH"

# Set working directory
WORKDIR /etc

# Start rsyslog in foreground
CMD ["rsyslogd", "-n"]

