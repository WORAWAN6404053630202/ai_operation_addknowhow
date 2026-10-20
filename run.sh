#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/run.sh
#!/bin/bash
# 2026-10 fix (Infra2-L5): removed `export $(grep -v '^#' env.properties | xargs)`.
# It word-split every value on whitespace (would have silently corrupted any
# secret or setting containing a space — see the LANGSMITH_PROJECT trailing-
# whitespace bug, Infra2-L3) and was dead weight besides: every compose file
# below (dev/staging/prod) already declares its own `env_file:` pointing at
# env.properties, so Compose loads the real values into each container
# directly — this line only ever polluted the host shell running `run.sh`.

if [ "$1" = "prod" ]; then
    # Start the service in production mode
    docker-compose -f ./docker-compose.prod.yaml up --build
elif [ "$1" = "sg" ]; then
    # Start the service in nolive mode
    docker-compose -f ./docker-compose.staging.yaml up --build
else
    # Start the service in development mode
    docker-compose -f ./docker-compose.dev.yaml up $1 $2
fi