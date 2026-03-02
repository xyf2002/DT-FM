import asyncio
import itertools
import time
import uuid

import aio_pika
import numpy as np
import torch

from morphling._Msg import (
    MatMulRequestMessage,
    MatMulResponseMessage,
    _custom_matmul,
)
from morphling.common.custom_logging import EventTimeLogger
from morphling.common.decorators import timeit_decorator

from .base import BaseBackend  # , MatMulRequestMessage, MatMulResponseMessage


class RabbitMQBackend(BaseBackend):
    def __init__(self, loop, host="amqp://localhost/", block_size=32):
        self.host = host
        self.loop = loop
        self.connection = None
        self.channel = None
        self.block_size = block_size

        self.current_time = 0

    async def connect(self):
        self.connection = await aio_pika.connect_robust(
            self.host, loop=self.loop
        )
        self.channel = await self.connection.channel()

        await self.channel.set_qos(prefetch_count=1)
        # Declare the queues and keep references to them
        self.request_queue = await self.channel.declare_queue(
            "mm_request_queue", durable=False
        )

        self.response_queue = await self.channel.declare_queue(
            "mm_response_queue", durable=False
        )

        # Set up the consumer using the correct method
        await self.response_queue.consume(self.on_response)

        self.timer_sync_request_exchange = await self.channel.declare_exchange(
            "timer_sync_request",
            aio_pika.ExchangeType.FANOUT,
        )
        self.timer_sync_set_exchange = await self.channel.declare_exchange(
            "timer_sync_set",
            aio_pika.ExchangeType.FANOUT,
        )

        # await self.channel.exchange_declare(exchange='timer_sync_request', exchange_type='fanout')
        self.timer_sync_response_queue = await self.channel.declare_queue(
            "timer_sync_response", durable=False
        )
        await self.timer_sync_response_queue.consume(self.on_timer_rsp)
        # await self.channel.exchange_declare(exchange='timer_sync_set', exchange_type='fanout')

        return self

    async def request_timer_sync(self):
        await self.timer_sync_request_exchange.publish(
            message=aio_pika.Message(
                b"",
                delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
            ),
            routing_key="",
        )

    async def on_timer_rsp(self, message: aio_pika.IncomingMessage):
        async with message.process():
            # interp the message as float
            elapsed = float(message.body.decode())

            self.current_time = max(self.current_time, elapsed)

            # send to timer_sync_set
            await self.timer_sync_set_exchange.publish(
                message=aio_pika.Message(
                    body=str(self.current_time).encode(),
                    delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                ),
                routing_key="",
            )

    # def connect(self):
    #     self.connection = pika.BlockingConnection(pika.URLParameters(self.host))
    #     self.channel = self.connection.channel()

    #     self.channel.queue_declare(queue="mm_request_queue", durable=False)
    #     self.channel.queue_declare(queue="mm_response_queue", durable=False)

    #     self.channel.basic_qos(prefetch_count=1)
    #     self.channel.basic_consume(queue="mm_response_queue", on_message_callback=self.on_response, auto_ack=True)

    #     return self

    @timeit_decorator
    async def on_response(self, message: aio_pika.IncomingMessage):
        async with message.process():
            response = MatMulResponseMessage()
            response.Deserialize(message.body)
            # await response.async_deserialize(message.body)

            r, c = response.row, response.col
            ld = response.ld
            # print(f"Received block ({r}, {c}, {self.c.device=} {response.result.device=})")

            offset_r = r * self.block_size
            offset_c = c * self.block_size

            size_r = response.mat.size(0)
            size_c = response.mat.size(1)

            assert not torch.allclose(
                response.mat, torch.zeros_like(response.mat)
            ), f"Received block ({r}, {c}) is zero"
            # print(f"Received block ({r}, {c})", response.mat)
            if ld:
                self.c[
                    ld,
                    offset_r : offset_r + size_r,
                    offset_c : offset_c + size_c,
                ] = response.mat
            else:
                self.c[
                    offset_r : offset_r + size_r,
                    offset_c : offset_c + size_c,
                ] = response.mat

            if torch.isnan(self.c).sum() == 0:
                self.finished = True

            # ack the message
            # self.channel.basic_ack(delivery_tag=method.delivery_tag)

    def create_output_matrix(self, mat_a: torch.Tensor, mat_b: torch.Tensor):
        # check matrix number of dimensions:
        # 1) mat_a and mat_b are 2D matrices
        # 2) mat_a > 2D and mat_b is 2D
        # 3) mat_a > 2D and mat_b > 2D

        a_shape = mat_a.size()
        b_shape = mat_b.size()

        print("create_output_matrix", a_shape, b_shape)

        in_dim = a_shape[-2]
        out_dim = b_shape[-1]

        # assert (
        #     in_dim % self.block_size == 0
        # ), f"Input dimension {in_dim} must be divisible by block size {self.block_size}"
        # assert (
        #     out_dim % self.block_size == 0
        # ), f"Output dimension {out_dim} must be divisible by block size {self.block_size}"

        if len(a_shape) == 2 and len(b_shape) == 2:
            c_shape = (in_dim, out_dim)
        elif len(a_shape) > 2 and len(b_shape) == 2:
            c_shape = a_shape[:-2] + (in_dim, out_dim)
        else:
            lda_shape = a_shape[:-2]
            ldb_shape = b_shape[:-2]
            # ldx dim must be the same
            assert lda_shape == ldb_shape, (
                f"Input dimensions {lda_shape} and {ldb_shape} must be the same"
            )
            c_shape = a_shape[:-2] + (in_dim, out_dim)

        self.c = torch.empty(c_shape, device="cpu")
        self.c[:] = float("nan")
        self.finished = False
        print(f"Created output {c_shape}")

        return in_dim, out_dim

    @timeit_decorator
    async def single_matmul_call(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor, r, c, ld
    ):
        request = MatMulRequestMessage()
        request.row = r
        request.col = c
        request.ld = ld

        # request.set(mat_a, mat_b, self.block_size)

        # offset_r = r * self.block_size
        # offset_c = c * self.block_size

        # size_r = min(self.block_size, mat_a.shape[-2] - r * self.block_size)
        # size_c = min(self.block_size, mat_b.shape[-1] - c * self.block_size)

        # request.mat = [mat_a[offset_r : offset_r + size_r], mat_b[:, offset_c : offset_c + size_c]]

        message = request.Serialize()
        # request.set(mat_a, mat_b, r, c, self.block_size, ld)
        # message = await request.async_serialize()
        corr_id = str(uuid.uuid4())

        # print(f"Sending block ({r}, {c})")
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=message,
                reply_to="mm_response_queue",
                correlation_id=corr_id,
                delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
            ),
            routing_key="mm_request_queue",
        )

    @timeit_decorator
    async def call(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        r: int,
        c: int,
    ):
        # check matrix number of dimensions:
        # 1) mat_a and mat_b are 2D matrices
        # 2) mat_a > 2D and mat_b is 2D
        # 3) mat_a > 2D and mat_b > 2D

        if len(mat_a.size()) == 2 and len(mat_b.size()) == 2:
            # print("call 2D-2D", mat_a.size(), mat_b.size())
            await self.single_matmul_call(mat_a, mat_b, r, c, [])

        elif len(mat_a.size()) > 2 and len(mat_b.size()) == 2:
            for i in range(mat_a.size(0)):
                # print("call >2D-2D", mat_a[i].size(), mat_b.size())
                await self.single_matmul_call(mat_a[i], mat_b, r, c, [i])
        else:
            ld = mat_a.size()[:-2]
            # get all combinations of indices for the leading dimensions
            ld_combinations = list(itertools.product(*[range(i) for i in ld]))
            for ld in ld_combinations:
                # print("call >2D->2D", ld, mat_a[ld].size(), mat_b[ld].size())
                await self.single_matmul_call(mat_a[ld], mat_b[ld], r, c, ld)

    @timeit_decorator
    async def dispatch_matmul(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor
    ) -> torch.Tensor:
        a_shape = mat_a.size()
        b_shape = mat_b.size()
        in_dim = mat_a.size(-2)
        out_dim = mat_b.size(-1)
        # padded = False
        # if not (in_dim % self.block_size == 0 and out_dim % self.block_size == 0):
        #     in_dim = in_dim + self.block_size - in_dim % self.block_size
        #     out_dim = out_dim + self.block_size - out_dim % self.block_size
        #     print(f"Padding input dimensions to {in_dim}x{out_dim}")

        #     padded = True

        #     # pad the matrices
        #     mat_a = torch.cat([mat_a, torch.zeros(a_shape[:-2] + (in_dim - a_shape[-2], a_shape[-1]))], dim=-2)
        #     mat_b = torch.cat([mat_b, torch.zeros(b_shape[:-2] + (b_shape[-2], out_dim - b_shape[-1]))], dim=-1)

        in_dim, out_dim = self.create_output_matrix(mat_a, mat_b)
        print(f"Dispatching {in_dim}x{out_dim} matrix multiplication")

        n_rows = in_dim // self.block_size + (in_dim % self.block_size > 0)
        n_cols = out_dim // self.block_size + (out_dim % self.block_size > 0)
        for r in range(n_rows):
            for c in range(n_cols):
                await self.call(mat_a, mat_b, r, c)

        while not self.finished:
            await asyncio.sleep(0.1)

        await self.request_timer_sync()

        # if padded:
        #     return self.c[:, :a_shape[-2], :b_shape[-1]]
        assert not torch.allclose(self.c, torch.zeros_like(self.c)), (
            f"Result is zero"
        )
        return self.c

    @timeit_decorator
    def sync_dispatch_matmul(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor
    ) -> torch.Tensor:
        print(
            "sync_dispatch_matmul, mat_a",
            mat_a.device,
            mat_a.size(),
            mat_a.dtype,
            ", mat_b",
            mat_b.device,
            mat_b.size(),
            mat_b.dtype,
        )
        self.loop.run_until_complete(self.dispatch_matmul(mat_a, mat_b))
        return self.c


class RabbitMQWorker:
    def __init__(self, device_info, loop, host="amqp://localhost/"):
        self.device_info = device_info
        self.cid = device_info["uuid"]
        self.loop = loop
        self.host = host
        self.connection = None
        self.channel = None
        # self.emulation = emulation

        self.last_dl_event = str(uuid.uuid4())
        self.last_ul_event = str(uuid.uuid4())
        self.last_dev_event = str(uuid.uuid4())

        self.time_logger = EventTimeLogger(
            self.device_info["uuid"],
            [self.last_dl_event, self.last_ul_event, self.last_dev_event],
        )

    async def connect(self):
        self.connection = await aio_pika.connect_robust(
            self.host, loop=self.loop
        )
        self.channel = await self.connection.channel()

        # Declare queues
        self.request_queue = await self.channel.declare_queue(
            "mm_request_queue", durable=False
        )
        self.response_queue = await self.channel.declare_queue(
            "mm_response_queue", durable=False
        )

        # check if queue is declared
        print(f"Queue {self.request_queue.name} declared", flush=True)
        print(f"Queue {self.response_queue.name} declared", flush=True)

        # Set Quality of Service
        await self.channel.set_qos(prefetch_count=1)

        self.timer_sync_request_exchange = await self.channel.declare_exchange(
            "timer_sync_request",
            aio_pika.ExchangeType.FANOUT,
        )
        self.timer_sync_set_exchange = await self.channel.declare_exchange(
            "timer_sync_set",
            aio_pika.ExchangeType.FANOUT,
        )

        # Declaring queue
        self.timer_sync_request_queue = await self.channel.declare_queue(
            exclusive=True
        )
        await self.timer_sync_request_queue.bind(
            self.timer_sync_request_exchange
        )
        await self.timer_sync_request_queue.consume(self.on_timer_req)

        self.timer_sync_response_queue = await self.channel.declare_queue(
            "timer_sync_response", durable=False
        )
        await self.timer_sync_response_queue.consume(self.on_timer_req)

        self.timer_sync_set_exchange = await self.channel.declare_exchange(
            "timer_sync_set",
            aio_pika.ExchangeType.FANOUT,
        )
        self.timer_sync_set_queue = await self.channel.declare_queue(
            exclusive=True
        )
        await self.timer_sync_set_queue.bind(self.timer_sync_set_exchange)
        await self.timer_sync_set_queue.consume(self.on_timer_set)

        print(f" [x] {self.cid}: connected to RabbitMQ", flush=True)

    async def on_timer_req(self, message: aio_pika.IncomingMessage):
        async with message.process():
            await self.channel.default_exchange.publish(
                message=aio_pika.Message(
                    body=str(self.time_logger.max_time()).encode(),
                    delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                ),
                routing_key="timer_sync_response",
            )

    async def on_timer_set(self, message: aio_pika.IncomingMessage):
        async with message.process():
            # print("on_timer_set")
            elapsed = float(message.body.decode())
            self.time_logger.set_time(elapsed)

    async def start_consuming(self):
        await self.request_queue.consume(self.on_request)
        print(f" [x] {self.cid}: Waiting for RPC requests")
        await asyncio.Future()  # Run forever

    @timeit_decorator
    async def on_request(self, message: aio_pika.IncomingMessage):
        async with message.process():
            print(f"[x] {self.cid}: Received request", flush=True)
            request = MatMulRequestMessage()
            request.Deserialize(message.body)
            # await request.async_deserialize(message.body)

            print(
                f"[x] {self.cid}: Received request row={request.row} col={request.col} ld={request.ld}",
                flush=True,
            )

            a_shape = request.mat_shape[0]
            b_shape = request.mat_shape[1]

            # assert torch.sum(mat_a) != 0, f"Received block row={request.row} col={request.col} is zero"
            # assert torch.sum(mat_b) != 0, f"Received block row={request.row} col={request.col} is zero"

            in_dim = a_shape.shape[0]
            h_dim = a_shape.shape[1]
            out_dim = b_shape.shape[1]

            # print(f"[x] {self.cid}: {mat_a.shape=} {mat_b.shape=}", flush=True)

            # bytes of tensors deviced by bandwidth + latency
            dl_elapsed = (
                in_dim * h_dim * 4 + h_dim * out_dim * 4
            ) / self.device_info["dl_bw"]
            dl_event_id = str(uuid.uuid4())
            self.time_logger.record(
                float(dl_elapsed), dl_event_id, [self.last_dl_event]
            )
            self.last_dl_event = dl_event_id

            # print("mat_a", mat_a)
            # print("mat_b", mat_b)

            # result = self.matmul(mat_a, mat_b)
            result = _custom_matmul(request, 0)
            # result = torch.mm(mat_a.to("cuda:0"), mat_b.to("cuda:0")).to("cpu")
            # result = torch.mm(mat_a, mat_b)
            dev_elapsed = (
                2 * in_dim * h_dim * out_dim / self.device_info["flops"]
            )
            dev_event_id = str(uuid.uuid4())
            self.time_logger.record(
                float(dev_elapsed),
                dev_event_id,
                [self.last_dev_event, self.last_dl_event],
            )
            self.last_dev_event = dev_event_id

            assert not torch.allclose(result, torch.zeros_like(result)), (
                f"Computed block row={request.row} col={request.col} is zero"
            )

            print(
                f"[x] {self.cid}: Computed block row={request.row} col={request.col} ld={request.ld}",
                flush=True,
            )

            response = MatMulResponseMessage()
            response.row = request.row
            response.col = request.col
            response.ld = request.ld
            response.mat = result
            message_body = response.Serialize()
            # response.set(result, request.r, request.c, request.ld)
            # message_body = await response.async_serialize()
            print(
                f"[x] {self.cid}: Serialized response row={response.row} col={response.col} ld={response.ld}",
                flush=True,
            )

            ul_elapsed = (
                response.mat.numel()
                * response.mat.element_size()
                / self.device_info["ul_bw"]
            )
            ul_event_id = str(uuid.uuid4())
            self.time_logger.record(
                float(ul_elapsed),
                ul_event_id,
                [self.last_ul_event, self.last_dev_event],
            )
            self.last_ul_event = ul_event_id

            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    correlation_id=message.correlation_id,
                    delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                ),
                routing_key=message.reply_to,
            )
            print(
                f"[x] {self.cid}: Sent response row={response.row} col={response.col} ld={response.ld}"
            )

    @timeit_decorator
    def matmul(self, a, b):
        # if self.emulation:
        #     # Emulate the device
        #     return torch.mm(a, b).to("cpu")

        # Check for MPS or CUDA availability and use the appropriate device
        device = "mps" if torch.backends.mps.is_available() else "cuda:0"
        # print(f"{a.dtype=} {b.dtype=} {a.device=} {b.device=} {device=}", flush=True)

        # print(f"{a.data_ptr()=}, {b.data_ptr()=}", flush=True)
        # return torch.mm(a, b)
        a_d = a.to(device)
        # print(f"{a_d.data_ptr()=}", flush=True)

        # print(f"{b=}", flush=True)
        b_d = b.to(device)
        # print(f"{b_d.data_ptr()=}", flush=True)

        # print(f"{a.dtype=} {b.dtype=} {a.device=} {b.device=} {device=}", flush=True)
        # print("matmul", a.device, b.device)
        c = torch.mm(a_d, b_d).to("cpu")
        # print(f"{c.dtype=} {c.device=}", flush=True)
        return c
