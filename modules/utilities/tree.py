class EmbeddingNode():
    def __init__(self, we, node):
        self.text = node.text.lower()
        self.dep = node.dep_
        self.pos = node.pos_
        self.representation = None
        self.embedding = we[self.text] if self.text in we else we['unk']
        self.chidren = [EmbeddingNode(we, c) for c in node.children]
    
    def subj(self):
        for c in self.chidren:
            if c.dep == "nsubj" or c.dep == "csubj" or c.dep == "nsubjpass" or c.dep == "csubjpass":
                return c
    
    def children_width(self, node):
        total_width = 0
        for c in node.chidren:
            total_width += len(c.text) + 2 + self.children_width(c)

        return total_width
    
    def width(self, node):
        return len(node.text) + 2 + self.children_width(node)

    def print_this_row(self, total_len, nodes):
        if not nodes:
            return ""
        next_row = []
        s = ' ' * total_len
        for n, i in nodes:
            l = len(n.text)+2
            s = s[:i] + '(' + n.text + ')' + s[i+l+1:]
            
            total_c_len = 0
            for c in n.chidren:
                c_ind = i + l + total_c_len
                next_row.append((c,c_ind))
                total_c_len += self.width(c)
        
        if next_row:
            s += '\n' + self.print_this_row(total_len, next_row)

        return s
    
    def __str__(self):
        return self.print_this_row(self.width(self), [(self, 0)])
