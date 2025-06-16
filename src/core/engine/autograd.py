from __future__ import annotations
import numpy as np

class Node:
    def __init__(self, tensor, _children=(), _op=''):
        self.tensor = tensor
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other: Node) -> Node:
        out = Node(self.tensor + other.tensor, (self, other), '+')

        if out.tensor.requires_grad:
            def _backward():
                if self.tensor.requires_grad:
                    self.tensor.grad += 1 * out.tensor.grad
                if other.tensor.requires_grad:
                    other.tensor.grad += 1 * out.tensor.grad
            out._backward = _backward
        
        return out
    
    def __sub__(self, other: Node) -> Node:
        out = Node(self.tensor - other.tensor, (self, other), '-')
        
        if out.tensor.requires_grad:
            def _backward():
                if self.tensor.requires_grad:
                    self.tensor.grad += 1 * out.tensor.grad
                if other.tensor.requires_grad:
                    other.tensor.grad -= 1 * out.tensor.grad
            out._backward = _backward
        
        return out
    
    def __mul__(self, scalar: np.float16) -> Node:
        out = Node(self.tensor * scalar, (self,), '*')
        
        if out.tensor.requires_grad and self.tensor.requires_grad:
            def _backward():
                self.tensor.grad += scalar * out.tensor.grad
            out._backward = _backward
        
        return out
    
    def __matmul__(self, other: Node) -> Node:
        out = Node(self.tensor @ other.tensor, (self, other), '@')
        
        if out.tensor.requires_grad:
            def _backward():
                if self.tensor.requires_grad:
                    self.tensor.grad += out.tensor.grad @ other.tensor.data.T
                if other.tensor.requires_grad:
                    other.tensor.grad += self.tensor.data.T @ out.tensor.grad
            out._backward = _backward
        
        return out
    
    def backward(self):
        visited = set()
        graph = []
        
        def build_graph(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_graph(child)
                graph.append(node)
        
        build_graph(self)
        self.tensor.grad = np.ones_like(self.tensor.data)
        

        for node in reversed(graph):
            node._backward()
    
    def zero_grad(self):
        self.tensor.grad = np.zeros_like(self.tensor.data)
        for child in self._prev:
            child.zero_grad()