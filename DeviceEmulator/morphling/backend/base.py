from abc import ABC, abstractmethod
from typing import List

import torch

import morphling.proto.morphling_pb2 as morphling_pb2
from morphling.common.decorators import timeit_decorator


class BaseBackend(ABC):
    @abstractmethod
    def dispatch_matmul(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor, block_size: int
    ):
        pass


import io

import torch


class MatMulRequestMessage:
    msg = morphling_pb2.MatMulRequestMessage()

    def set(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        r: int,
        c: int,
        block_size: int,
        ld: List[int] = list(),
    ):
        self.a = mat_a[r * block_size : (r + 1) * block_size, :]
        self.b = mat_b[:, c * block_size : (c + 1) * block_size]

        self.r = r
        self.c = c
        self.ld = ld

        print(
            "MatMulRequestMessage set",
            self.r,
            self.c,
            self.ld,
            self.a.shape,
            self.b.shape,
        )

    @timeit_decorator
    def serialize(self) -> bytes:
        a_bytes = io.BytesIO()
        torch.save(self.a, a_bytes)
        a_bytes = a_bytes.getvalue()

        b_bytes = io.BytesIO()
        torch.save(self.b, b_bytes)
        b_bytes = b_bytes.getvalue()

        self.msg.row = self.r
        self.msg.col = self.c
        self.msg.ld[:] = self.ld
        self.msg.mat[:] = [a_bytes, b_bytes]
        # self.msg.mat.append(b_bytes)

        return self.msg.SerializeToString()

        # a_bytes = io.BytesIO()
        # torch.save(self.a, a_bytes)
        # a_bytes = a_bytes.getvalue()
        # a_byte_len = len(a_bytes)

        # b_bytes = io.BytesIO()
        # torch.save(self.b, b_bytes)
        # b_bytes = b_bytes.getvalue()
        # b_byte_len = len(b_bytes)

        # self.msg.set

        # # pickle self.ld
        # ld_bytes = b''.join([d.to_bytes(8, byteorder="big") for d in self.ld])
        # ld_byte_len = len(self.ld)

        # print("serialize ld", self.ld, ld_byte_len, ld_bytes)

        # body_bytes = (
        #     self.r.to_bytes(8, byteorder="big")
        #     + self.c.to_bytes(8, byteorder="big")
        #     + ld_byte_len.to_bytes(8, byteorder="big")
        #     + ld_bytes
        #     + a_byte_len.to_bytes(8, byteorder="big")
        #     + b_byte_len.to_bytes(8, byteorder="big")
        #     + a_bytes
        #     + b_bytes
        # )

        # return body_bytes

    @timeit_decorator
    def deserialize(self, body: bytes):
        self.msg.ParseFromString(body)

        self.r = self.msg.row
        self.c = self.msg.col
        self.ld = self.msg.ld

        a_bytes = self.msg.mat[0]
        b_bytes = self.msg.mat[1]

        self.a = torch.load(io.BytesIO(a_bytes))
        self.b = torch.load(io.BytesIO(b_bytes))

        # self.r = int.from_bytes(body[:8], byteorder="big")
        # self.c = int.from_bytes(body[8:16], byteorder="big")

        # ld_size = int.from_bytes(body[16:24], byteorder="big")
        # ld = [int.from_bytes(body[24+i*8:32+i*8], byteorder="big") for i in range(ld_size)]
        # ld_size = ld_size * 8
        # ld_bytes = body[24:24+ld_size]

        # print("deserialize ld", ld, ld_size, ld_bytes)

        # a_size = int.from_bytes(body[24+ld_size:32+ld_size], byteorder="big")
        # a = body[32+ld_size:32+ld_size+a_size]

        # b_size = int.from_bytes(body[32+ld_size+a_size:40+ld_size+a_size], byteorder="big")
        # b = body[40+ld_size+a_size:40+ld_size+a_size+b_size]

        # self.a = torch.load(io.BytesIO(a))
        # self.b = torch.load(io.BytesIO(b))
        # self.ld = ld

    # all functions has async wrapper

    async def async_serialize(self) -> bytes:
        return self.serialize()

    async def async_deserialize(self, body: bytes):
        self.deserialize(body)


class MatMulResponseMessage:
    msg = morphling_pb2.MatMulResponseMessage()

    def set(self, result: torch.Tensor, r: int, c: int, ld: List[int] = list()):
        self.result = result
        self.r = r
        self.c = c
        self.ld = ld

    @timeit_decorator
    def serialize(self) -> bytes:
        result_bytes = io.BytesIO()
        torch.save(self.result, result_bytes)
        result_bytes = result_bytes.getvalue()

        self.msg.row = self.r
        self.msg.col = self.c
        self.msg.ld[:] = self.ld
        self.msg.mat = result_bytes

        return self.msg.SerializeToString()

        # result_bytes = io.BytesIO()
        # torch.save(self.result, result_bytes)
        # result_bytes = result_bytes.getvalue()
        # result_byte_len = len(result_bytes)

        # # pickle self.ld
        # ld_bytes = b''.join([d.to_bytes(8, byteorder="big") for d in self.ld])
        # ld_byte_len = len(self.ld)

        # body_bytes = (
        #     self.r.to_bytes(8, byteorder="big")
        #     + self.c.to_bytes(8, byteorder="big")
        #     + ld_byte_len.to_bytes(8, byteorder="big")
        #     + ld_bytes
        #     + result_byte_len.to_bytes(8, byteorder="big")
        #     + result_bytes
        # )

        # return body_bytes

    @timeit_decorator
    def deserialize(self, body: bytes):
        self.msg.ParseFromString(body)

        self.r = self.msg.row
        self.c = self.msg.col

        self.ld = self.msg.ld

        result_bytes = self.msg.mat
        self.result = torch.load(io.BytesIO(result_bytes))

        # self.r = int.from_bytes(body[:8], byteorder="big")
        # self.c = int.from_bytes(body[8:16], byteorder="big")

        # ld_size = int.from_bytes(body[16:24], byteorder="big")
        # ld = [int.from_bytes(body[24+i*8:32+i*8], byteorder="big") for i in range(ld_size)]
        # ld_size = ld_size * 8

        # result_size = int.from_bytes(body[24+ld_size:32+ld_size], byteorder="big")
        # result = body[32+ld_size:32+ld_size+result_size]

        # self.result = torch.load(io.BytesIO(result))
        # self.ld = ld

    async def async_serialize(self) -> bytes:
        return self.serialize()

    async def async_deserialize(self, body: bytes):
        return self.deserialize(body)
