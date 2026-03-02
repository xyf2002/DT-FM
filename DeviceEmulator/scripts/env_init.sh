pkill -f "morphling_device"

# CONF_PWD="$PWD/../config"

# # if docker contrianer mosquitto is running, stop it
# if [ "$(docker ps -q -f name=mosquitto)" ]; then
#     docker stop mosquitto
# fi


# docker run -dit --rm -p 1883:1883 --name mosquitto -v "$CONF_PWD/mosquitto/config:/mosquitto/config" -v "$CONF_PWD/mosquitto/data:/mosquitto/data" -v "$CONF_PWD/mosquitto/log:/mosquitto/log" eclipse-mosquitto
# sleep 5

# # if docker contrianer emqx is running, stop it
# if [ "$(docker ps -q -f name=emqx)" ]; then
#     docker stop emqx
# fi

# docker run -dit --rm --name emqx -p 1883:1883 -p 8083:8083 -p 8084:8084 -p 8883:8883 -p 18083:18083 \
#     -v "$CONF_PWD/emqx/emqx.conf:/opt/emqx/etc/emqx.conf" -e ERL_AFLAGS="+S 28:28" emqx/emqx:latest
# sleep 5

# # if docker contrianer rabbitmq is running, stop it
# if [ "$(docker ps -q -f name=rabbitmq)" ]; then
#     docker stop rabbitmq
# fi


# # start rabbitmq container, set MAX_MSG_SIZE to 128MB
# docker run -dit --rm --name rabbitmq -p 5672:5672 -p 1883:1883 -p 15672:15672 rabbitmq:4.0-management
# sleep 5

# # run command in rabbitmq container "rabbitmq-plugins enable rabbitmq_mqtt"
# docker exec rabbitmq rabbitmq-plugins enable rabbitmq_mqtt

# # if docker contrianer redis is running, stop it
# if [ "$(docker ps -q -f name=redis)" ]; then
#     docker stop redis
# fi

# docker run -dit --rm --name redis -p 6379:6379 redis
# sleep 5
