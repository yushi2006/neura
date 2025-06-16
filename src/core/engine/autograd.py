from __future__ import annotations
import numpy as np

class Node:
    def __init__(self, tensor, _children=(), _op=''):
        self.tensor = tensor
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other: Node) -> Node:
        out = Node(self.tensor + other.tensor)

        if out.tensor.requires_grad:
            out._prev = (self, other)
            out._op = '+'

            def _backward():
                self.tensor.grad += out.tensor.grad
                other.tensor.grad += out.tensor.grad

            out._backward = _backward

        return out
    
    def __sub__(self, other: Node) -> Node:
        out = Node(self.data - other.data)

        if out.tensor.requires_grad:
            out._prev = (self, other)
            out._op = '-'

            def _backward():
                self.tensor.grad += out.tensor.grad
                other.tensor.grad -= out.tensor.grad

            out._backward = _backward

        return out
    
    def __mul__(self, scalar: np.float16) -> Node:
        out = Node(self.tensor * scalar)

        if out.tensor.requires_grad:
            out._prev = (self,)
            out._op = '*'

            def _backward():
                self.tensor.grad += scalar * out.tensor.grad

            out._backward = _backward

        return out
    
    def __matmul__(self, other: Node) -> Node:
        out = Node(self.data @ other.data)

        if out.tensor.requires_grad:
            out._prev = (self, other)
            out._op = '@'

            def _backward():
                self.tensor.grad += other.tensor.T @ out.grad
                other.tensor.grad += self.tensor @ out.grad.T
            
            out._backward = _backward

        return out
    
    def backward(self):
        graph = []
        visited = set()
        def build_graph(node):
            visited.add(node)
            for child in node._prev:
                build_graph(child)
            graph.append(node)
        build_graph(self)
        self.grad = np.ones_like(self.data.shape)
        for node in reversed(graph):
            node._backward()
    
    def zero_grad(self):
        self.tensor.grad = np.zeros_like(self.data)
        for child in self._prev:
            child.zero_grad()
