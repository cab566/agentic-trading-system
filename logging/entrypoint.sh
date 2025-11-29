#!/bin/bash

# Create log directories
mkdir -p /fluentd/log/aggregated

# Start Fluentd
exec fluentd -c /fluentd/etc/fluent.conf -v