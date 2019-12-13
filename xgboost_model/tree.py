import treelib

class Tree():
    def __init__(self, init_value):
        self.tree = treelib.Tree()
        self.tree.create_node(identifier=1,
                              data=init_value)

    def set_value(self, nid, value):
        node = self.tree.get_node(nid)
        node.data = value

    def get_value(self, nid):
        try:
            return self.tree.get_node(nid).data
        except:
            Exception("An attempt to obtain non-existing node occured.".format(nid))

    def get_split(self, nid, left_value, right_value):
        self.tree.create_node(identifier=2 * nid,
                              parent=nid,
                              data=left_value)
        self.tree.create_node(identifier=2 * nid + 1,
                              parent=nid,
                              data=right_value)
