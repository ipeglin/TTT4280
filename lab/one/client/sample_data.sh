#!/bin/bash

# Define SSH connection details
SSH_USER="nelip"
SSH_HOST="raspberrypi.local"
REMOTE_DIR="TTT4280/server/sampler"


SSH_COMMAND="cd $REMOTE_DIR && sudo ./adc_sampler 31250 && ls -t | grep 'out' | head -n 1"


# Execute SSH command and retrieve the most recent file name
REMOTE_FILE=$(ssh $SSH_USER@$SSH_HOST "$SSH_COMMAND")

SLEEP 5

# Print the remote file to check if it's correctly obtained
echo "Remote File: $REMOTE_FILE"

# Check if the remote file is empty or doesn't exist
if [ -z "$REMOTE_FILE" ]; then
    echo "No files found matching the criteria."
    exit 1
fi

# Fetch the most recent file back to the local machine
scp $SSH_USER@$SSH_HOST:$REMOTE_DIR/$REMOTE_FILE .

echo "File $REMOTE_FILE has been downloaded from $SSH_HOST"
