pkill -f "test_backend_mqtt"

# if docker contrianer rabbitmq is running, stop it
if [ "$(docker ps -q -f name=rabbitmq)" ]; then
    docker stop rabbitmq
fi


# start rabbitmq container, set MAX_MSG_SIZE to 128MB
docker run -dit --rm --name rabbitmq -p 5672:5672 -p 1883:1883 -p 15672:15672 rabbitmq:4.0-management
sleep 5

# run command in rabbitmq container "rabbitmq-plugins enable rabbitmq_mqtt"
docker exec rabbitmq rabbitmq-plugins enable rabbitmq_mqtt

# if docker contrianer redis is running, stop it
if [ "$(docker ps -q -f name=redis)" ]; then
    docker stop redis
fi

docker run -dit --rm --name redis -p 6379:6379 redis
sleep 5

NUM_DEVICES=10
for i in $(seq 1 ${NUM_DEVICES}); do
    device_id=$(($i-1))
    SPDLOG_LEVEL="debug" python3 tests/python/test_backend_mqtt.py --type "worker" --num_devices $NUM_DEVICES --device_id $device_id &
done
sleep 5


NUM_MOSQ=10 SPDLOG_LEVEL="debug" python3 tests/python/test_backend_mqtt.py --type "server" --num_devices $NUM_DEVICES --device_id -1
# NUM_MOSQ=10 SPDLOG_LEVEL="debug" perf record -F 99 -a -g -- python3 tests/python/test_backend_mqtt.py --type "server" --num_devices $NUM_DEVICES --device_id -1 > out.perf
# perf script > out.perf

# ${HOME}/FlameGraph/stackcollapse-perf.pl out.perf > out.folded
# ${HOME}/FlameGraph/flamegraph.pl out.folded > kernel.svg
