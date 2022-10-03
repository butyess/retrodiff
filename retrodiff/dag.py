from collections import deque


class Node:
    def __init__(self, function=None, input_nodes=[]):
        self.function = function
        self.input_nodes = input_nodes
        self.is_input_node = len(input_nodes) == 0
        self._value = None
        self.grad = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if not self.is_input_node:
            raise ValueError("Setting a value to a non-input node")
        self._value = v

    def replace_node(self, orig, repl):
        queue = deque((self,))
        visited = set()
        while len(queue) > 0:
            n = queue.popleft()
            for i, x in enumerate(n.input_nodes):
                if x == orig:
                    n.input_nodes[i] = repl
                else:
                    if x not in visited:
                        queue.append(x)
            visited.add(n)

    def forward(self):
        if self.is_input_node:
            return self._value
        ival = [n.forward() for n in self.input_nodes]
        self._value = self.function.forward(*ival)
        return self._value

    def backward(self, first_grad):
        order = self.topo_sort()
        self.grad = first_grad

        for node in order:
            node.update_input_grads()

    def clear(self):
        self._value = None
        for n in self.input_nodes:
            n.clear()
    
    def topo_sort(self):
        topo_order = [self]
        queue = deque(self.input_nodes)
        while len(queue) > 0:
            n = queue.popleft()
            if n not in topo_order:
                topo_order.append(n)
                queue.extend(n.input_nodes)
        return topo_order

    def update_input_grads(self):
        ival = [i._value for i in self.input_nodes]
        for i, n in enumerate(self.input_nodes):
            if n.grad is None:
                n.grad = self.function.backward(self.grad, i, *ival)
            else:
                # will `+=` always work? ∇ is linear, but...
                n.grad += self.function.backward(self.grad, i, *ival)

    def show_tree(self, indent=0):
        s = ' '*indent + str(self)
        for n in self.input_nodes:
            s += '\n' + n.show_tree(indent + 2)
        return s
    
    def __str__(self):
        if self.is_input_node:
            return f'<input node: val={self._value}, grad={self.grad}>'
        return f'<node: fn={type(self.function).__name__}, val={self._value}, ' + \
               f'grad={self.grad}, inputs={len(self.input_nodes)}>'

    __repr__ = __str__


class Function:
    def forward(self, *args):
        '''
        Computes the function.
        '''
        raise NotImplementedError

    def backward(self, grad, wrt, *values):
        '''
        Performs backward pass with respect to the parameter `wrt`.
        It returns dn/di * df/di, where df/di is the parameter `grad`,
        and df/di depends on `values` and `wrt`.
        '''
        raise NotImplementedError

    def __call__(self, *args):
        return Node(function=self, input_nodes=list(args))
