class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
        # Compare the new value with the parent node
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)

            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)

        else:
            self.data = data

    def print_node(self):
        # Print held data
        print(self.data)

    def print_children(self):
        # Print inorder
        if self.left:
            self.left.print_node()

        print(self.data)

        if self.right:
            self.right.print_node()
