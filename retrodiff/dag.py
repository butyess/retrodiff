class Dag:
    def __init__(self, input_nodes, output_node):
        self.output_node = output_node
        if set(input_nodes) != output_node.find_inputs():
            raise ValueError("Invalid input nodes")
        self.input_nodes = input_nodes

    def forward(self, input_values):
        assert len(input_values) == len(self.input_nodes)
        for node, val in zip(self.input_nodes, input_values):
            node.value = val
        return self.output_node.value

    def backward(self, first_grad):
        order = self.topo_sort(self.output_node)
        self.output_node.grad = first_grad

        for node in order:
            node.update_input_grads()
        return [n.grad for n in self.input_nodes]

    def topo_sort(self, node):
        topo_order = []
        queue = [node]
        while len(queue) > 0:
            n = queue.pop(0)
            if n not in topo_order:
                topo_order.append(n)
                for i in n.input_nodes:
                    queue.append(i)
        return topo_order

    def clear(self):
        for node in self.topo_sort(self.output_node):
            del node.value
            node.grad = None


class Node:
    def __init__(self, function=None, input_nodes=[]):
        self.function = function
        self.input_nodes = input_nodes
        self._value = None
        self.grad = None

    @property
    def value(self):
        if self._value is None:
            ival = [i.value for i in self.input_nodes]
            self._value = self.function.forward(*ival)
        return self._value

    @value.setter
    def value(self, value):
        assert len(self.input_nodes) == 0, "not an input node"
        self._value = value

    @value.deleter
    def value(self):
        self._value = None

    def find_inputs(self):
        if len(self.input_nodes) == 0:
            return set((self,))
        return set.union(*[n.find_inputs() for n in self.input_nodes])

    def update_input_grads(self):
        ival = [i.value for i in self.input_nodes]
        for i, n in enumerate(self.input_nodes):
            if n.grad is None:
                n.grad = self.function.backward(self.grad, i, *ival)
            else:
                # will `+=` always work? ∇ is linear, but...
                n.grad += self.function.backward(self.grad, i, *ival)


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
