class EmbeddingNode():
    def __init__(self, we, node):
        self.text = node.text.lower()
        self.dep = node.dep_
        self.representation = None
        self.embedding = we[self.text] if self.text in we else we['unk']
        self.children = [EmbeddingNode(we, c) for c in node.children]
