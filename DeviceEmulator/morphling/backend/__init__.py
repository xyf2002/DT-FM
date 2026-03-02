from morphling._Msg import (
    AMQPBackend,
    AMQPWorker,
    MQTTServer,
    MQTTWorker,
    ProxyCli,
    ProxySvr,
)

from .base import BaseBackend, MatMulRequestMessage, MatMulResponseMessage
from .rabbitmq import RabbitMQBackend, RabbitMQWorker


# auto backend from name
class AutoBackend:
    @classmethod
    def from_name(cls, name, *args, **kwargs):
        if name == "rabbitmq":
            print("Using RabbitMQ backend")
            return RabbitMQBackend(*args, **kwargs)
        elif name == "amqp":
            print("Using AMQP backend")
            return AMQPBackend(args[0], args[1])
        elif name == "mqtt":
            print("Using MQTT backend")
            return MQTTServer(args[0])
        elif name == "proxy":
            print("Using Proxy backend")
            return ProxySvr()
        else:
            raise ValueError(f"Unknown backend: {name}")


class AutoWorker:
    @classmethod
    def from_name(cls, name, *args, **kwargs):
        if name == "rabbitmq":
            print("Using RabbitMQ worker")
            return RabbitMQWorker(*args, **kwargs)
        elif name == "amqp":
            print("Using AMQP worker")
            return AMQPWorker(args[0], args[1])
        elif name == "mqtt":
            print("Using MQTT worker")
            return MQTTWorker(args[0])
        elif name == "proxy":
            print("Using Proxy worker")
            return ProxyCli()
        else:
            raise ValueError(f"Unknown worker: {name}")
