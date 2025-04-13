from collections import deque

class TreeNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.locked_by = -1
        self.is_locked = False
        self.childs = []
        self.parent = parent
        self.locked_descendents = set()

    def add_child(self, names):
        for name in names:
            self.childs.append(TreeNode(name, self))

class MAryTree:
    def __init__(self, root_name):
        self.root = TreeNode(root_name)
        self.name_to_node = {}

    def make_mary_tree(self, names, m):
        q = deque()
        k = 1
        n = len(names)
        q.append(self.root)

        while q:
            r = q.popleft()
            self.name_to_node[r.name] = r
            b = []
            for i in range(k, min(n, k + m)):
                b.append(names[i])
            r.add_child(b)
            for child in r.childs:
                q.append(child)
            k += len(b)

    def print_tree(self, r=None):
        if r is None:
            r = self.root
        if not r:
            return
        print(f"TreeNode -> {r.name} {r.locked_by}")
        print("Childs -> ")
        for child in r.childs:
            print("       ", child.name)
        print("Locked Descendants -> ")
        for child in r.locked_descendents:
            print("       ", child.name)
        for child in r.childs:
            self.print_tree(child)

    def update_parents(self, r, curr):
        while r:
            r.locked_descendents.add(curr)
            r = r.parent

    def lock(self, name, id):
        r = self.name_to_node.get(name)
        if r.is_locked or len(r.locked_descendents) > 0:
            return False

        par = r.parent
        while par:
            if par.is_locked:
                return False
            par = par.parent

        self.update_parents(r.parent, r)
        r.is_locked = True
        r.locked_by = id
        return True

    def unlock(self, name, id):
        r = self.name_to_node.get(name)
        if not r.is_locked or r.locked_by != id:
            return False

        par = r.parent
        while par:
            par.locked_descendents.discard(r)
            par = par.parent

        r.is_locked = False
        r.locked_by = -1
        return True

    def upgrade_lock(self, name, id):
        r = self.name_to_node.get(name)
        if r.is_locked or len(r.locked_descendents) == 0:
            return False

        for ld in r.locked_descendents:
            if ld.locked_by != id:
                return False

        par = r.parent
        while par:
            if par.is_locked:
                return False
            par = par.parent

        locked_descendants_copy = set(r.locked_descendents)
        for ld in locked_descendants_copy:
            self.unlock(ld.name, id)

        self.lock(name, id)
        return True


# Driver code equivalent
if __name__ == "__main__":
    n, m, t = map(int, input().split())
    names = input().split()
    tree = MAryTree(names[0])
    tree.make_mary_tree(names, m)

    for _ in range(t):
        op_type, name, id = input().split()
        id = int(id)
        if op_type == '1':
            print("true" if tree.lock(name, id) else "false")
        elif op_type == '2':
            print("true" if tree.unlock(name, id) else "false")
        elif op_type == '3':
            print("true" if tree.upgrade_lock(name, id) else "false")
-----------------------------------------------------------------------------
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

    def build_tree(self, names, m):
        from collections import deque
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

    def update_ancestors_on_lock(self, node):
        while node:
            node.locked_descendants.add(node)
            node = node.parent

    def update_ancestors_on_unlock(self, node):
        while node:
            node.locked_descendants.discard(node)
            node = node.parent

    def has_locked_ancestor(self, node):
        while node:
            if node.is_locked:
                return True
            node = node.parent
        return False

    def lock(self, name, user_id):
        node = self.name_to_node[name]
        if node.is_locked or node.locked_descendants:
            return False
        if self.has_locked_ancestor(node.parent):
            return False
        self.update_ancestors_on_lock(node.parent)
        node.is_locked = True
        node.locked_by = user_id
        return True

    def unlock(self, name, user_id):
        node = self.name_to_node[name]
        if not node.is_locked or node.locked_by != user_id:
            return False
        self.update_ancestors_on_unlock(node.parent)
        node.is_locked = False
        node.locked_by = -1
        return True

    def upgrade(self, name, user_id):
        node = self.name_to_node[name]
        if node.is_locked or not node.locked_descendants:
            return False
        if any(nd.locked_by != user_id for nd in node.locked_descendants):
            return False
        if self.has_locked_ancestor(node.parent):
            return False
        locked_nodes = list(node.locked_descendants)
        for desc in locked_nodes:
            self.unlock(desc.name, user_id)
        return self.lock(name, user_id)


# --- Example Usage ---
if __name__ == "__main__":
    n, m, q = map(int, input().split())
    node_names = input().split()
    tree = MAryTree(node_names[0])
    tree.build_tree(node_names, m)

    for _ in range(q):
        op_type, name, user_id = input().split()
        user_id = int(user_id)
        if op_type == "1":
            print("true" if tree.lock(name, user_id) else "false")
        elif op_type == "2":
            print("true" if tree.unlock(name, user_id) else "false")
        elif op_type == "3":
            print("true" if tree.upgrade(name, user_id) else "false")
--------------------------------------------------------------------------------

import sys
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


# --- Main Function for General Input ---
def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    n, m, q = map(int, data[0].split())
    names = data[1].split()
    queries = [tuple(data[i].split()) for i in range(2, 2 + q)]

    tree = MAryTree(names[0])
    tree.make_m_ary_tree(names, m)

    for query in queries:
        op_type, name, user_id = int(query[0]), query[1], int(query[2])
        if op_type == 1:
            print(tree.lock(name, user_id))
        elif op_type == 2:
            print(tree.unlock(name, user_id))
        elif op_type == 3:
            print(tree.upgrade(name, user_id))


# Example for running manually:
# if __name__ == "__main__":
#     main()
