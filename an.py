from collections import deque

class TreeNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.locked_by = -1
        self.is_locked = False
        self.children = []
        self.parent = parent
        self.locked_descendants = set()

    def add_children(self, names):
        for name in names:
            child = TreeNode(name, self)
            self.children.append(child)


class MAryTree:
    def __init__(self, root_name):
        self.root = TreeNode(root_name)
        self.name_to_node = {root_name: self.root}

    def make_m_ary_tree(self, names, m):
        queue = deque([self.root])
        idx = 1
        while queue and idx < len(names):
            node = queue.popleft()
            child_names = names[idx:idx + m]
            node.add_children(child_names)
            for child in node.children:
                self.name_to_node[child.name] = child
                queue.append(child)
            idx += m

    def update_ancestors_on_lock(self, node, current):
        while node:
            node.locked_descendants.add(current)
            node = node.parent

    def update_ancestors_on_unlock(self, node, current):
        while node:
            node.locked_descendants.discard(current)
            node = node.parent

    def lock(self, name, user_id):
        node = self.name_to_node[name]
        if node.is_locked or len(node.locked_descendants) > 0:
            return False
        parent = node.parent
        while parent:
            if parent.is_locked:
                return False
            parent = parent.parent
        self.update_ancestors_on_lock(node.parent, node)
        node.is_locked = True
        node.locked_by = user_id
        return True

    def unlock(self, name, user_id):
        node = self.name_to_node[name]
        if not node.is_locked or node.locked_by != user_id:
            return False
        self.update_ancestors_on_unlock(node.parent, node)
        node.is_locked = False
        node.locked_by = -1
        return True

    def upgrade(self, name, user_id):
        node = self.name_to_node[name]
        if node.is_locked or len(node.locked_descendants) == 0:
            return False
        for desc in node.locked_descendants:
            if desc.locked_by != user_id:
                return False
        parent = node.parent
        while parent:
            if parent.is_locked:
                return False
            parent = parent.parent
        to_unlock = list(node.locked_descendants)
        for desc in to_unlock:
            self.unlock(desc.name, user_id)
        return self.lock(name, user_id)



def main():
    n = int(input())
    m = int(input())
    q = int(input())
    
    names = [input().strip() for _ in range(n)]
    queries = [input().strip().split() for _ in range(q)]

    tree = MAryTree(names[0])
    tree.make_m_ary_tree(names, m)

    for query in queries:
        op_type = int(query[0])
        name = query[1]
        user_id = int(query[2])

        result = False
        if op_type == 1:
            result = tree.lock(name, user_id)
        elif op_type == 2:
            result = tree.unlock(name, user_id)
        elif op_type == 3:
            result = tree.upgrade(name, user_id)
        
        print(str(result).lower())  
if __name__ == "__main__":
    main()
