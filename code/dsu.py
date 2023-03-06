"""
Class to declare a disjoint set data structure
and its related class functions

"""
class DSU:
    def __init__(self,size):
        self._parent={}
        self._set_size={}
        self.size = size
        self.initialize()

    
    #Funciton to find the parent of a leaf recursively
    @property
    def parent(self,p):
        if self._parent[p]==p:
            return p
        else:
            return self.parent(self._parent[p])
    
    #Funciton to set parent of leaf i to the parent of leaf j 
    @parent.setter
    def set_parent(self,i,j):
        self._parent[i]=j

    
    #Function to initialize the DSU array
    def initialize(self):
        for i in range(self.size):
            self._parent[i]=i
            self._set_size[i]=1
    
    #Function to perform union of set i and j
    def union(self,i,j):
        if self.parent(i)==self.parent(j):
            pass
        else:
            p1 = self.parent(i)
            p2 = self.parent(j)

            #Set the parent of p1 to p2
            self._parent[p1]=p2

            #Set the set_size of p1 to p1+p2
            self._set_size[p2]=self._set_size[p1]+self._set_size[p2]
    
    
        
    