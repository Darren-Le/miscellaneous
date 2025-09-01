from collections import deque
class Node:
    def __init__(self, id, f, th, label, left=None, right=None):
        self.id = id
        self.f = f
        self.th = th
        self.label = label
        self.left = left
        self.right = right
        self.flag = False
        self.type = True # True means not leaves
        
n, m ,k = map(int, input().split())
nodes = [list(map(int, input().split())) for _ in range(n)]
samples = [list(map(int, input().split())) for _ in range(m)]
samples = [[sample[-1]] + sample[:-1] for sample in samples ]
def build_tree(nodes):
    l, r, f, th, label = nodes[0]
    root = Node(1, f, th, label)
    stack = deque([[root, l, r]])
    
    for i in range(1, len(nodes), 2):
        cur, l_, r_ = stack.popleft()
        l, r, f, th, label = nodes[i]
        left_node = Node(l_, f, th, label)
        cur.left = left_node
        if not (l == 0 and r == 0):
            stack.append([left_node, l, r])
        else:
            left_node.type = False
            
        if i + 1 < len(nodes):
            l, r, f, th, label = nodes[i + 1]
            right_node = Node(r_, f, th, label)
            cur.right = right_node
            if not (l == 0 and r == 0):
                stack.append([right_node, l, r])
            else:
                right_node.type = False
    
    return root

def print_tree(node):
    if not node:
        return
    print(node.id, node.f, node.th, node.label)
    print_tree(node.left)
    print_tree(node.right)
    
def predict(sample, root):
    cur = root
    while cur.left or cur.right:
        if sample[cur.f] <= cur.th:
            cur = cur.left
        else:
            cur = cur.right
    
    if cur.label == sample[0]:
        cur.flag = True
    return [cur.flag, cur]
        
root = build_tree(nodes)


for sample in samples:
    predict(sample, root)

def prune(node):
    if node == None:
        return True
    
    if node.type:
        res1 = prune(node.left)
        res2 = prune(node.right)
        if res1 and res2:
            node.type = False
            node.left = None
            node.right = None
    
    # if node.left and node.right:
    #     if node.left.flag == False and node.right.flag == False:
    #         node.left = None
    #         node.right = None
    # elif node.left:
    #     if node.left.flag == False:
    #         node.left = None
    # elif node.right:
    #     if node.right.flag == False:
    #         node.right = None
    
    

# print_tree(root)
