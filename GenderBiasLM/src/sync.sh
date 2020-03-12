#!/bin/bash

echo "Syncing to server"
rsync -avz --exclude-from '.syncignore' ./ LISA-FACT-20:~/
echo "Done Syncing"