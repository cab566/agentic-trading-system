#!/bin/bash
set -e

# PostgreSQL Slave Initialization Script
# This script sets up streaming replication from the master

echo "Initializing PostgreSQL slave..."

# Wait for master to be ready
echo "Waiting for master database to be ready..."
until pg_isready -h postgres-master -p 5432 -U postgres; do
  echo "Master not ready, waiting..."
  sleep 2
done

echo "Master is ready, setting up replication..."

# Stop PostgreSQL if running
pg_ctl -D "$PGDATA" -m fast -w stop || true

# Remove existing data directory contents
rm -rf "$PGDATA"/*

# Create base backup from master
echo "Creating base backup from master..."
PGPASSWORD="$POSTGRES_REPLICATION_PASSWORD" pg_basebackup \
    -h postgres-master \
    -D "$PGDATA" \
    -U replicator \
    -v \
    -P \
    -W \
    -R

# Create recovery configuration
cat > "$PGDATA/postgresql.conf" << EOF
# PostgreSQL Slave Configuration
hot_standby = on
max_connections = 100
shared_buffers = 128MB
effective_cache_size = 256MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Replication settings
hot_standby_feedback = on
max_standby_streaming_delay = 30s
max_standby_archive_delay = 30s

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_messages = info
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0
log_autovacuum_min_duration = 0
log_error_verbosity = default
EOF

# Set proper permissions
chown -R postgres:postgres "$PGDATA"
chmod 700 "$PGDATA"

echo "Slave initialization completed successfully!"
echo "Starting PostgreSQL slave..."

# Start PostgreSQL
exec postgres