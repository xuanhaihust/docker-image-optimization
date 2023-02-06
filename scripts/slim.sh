docker-slim build \
    --http-probe-start-wait=5  \
    --http-probe=false \
    --http-probe-cmd /health \
    --publish-exposed-ports \
    --expose 8080 --network host \
    --exec 'bash; ls' \
    classifier:v7

