import numpy as np

class Skeleton():
    def __init__(self, skl_type='h36m', joint_n=22):
        self.joint_n = joint_n
        self.get_bone(skl_type)
        self.get_skeleton()

    def get_bone(self, skl_type):
        self_link = [(i, i) for i in range(self.joint_n)]
        if skl_type == 'h36m':
            if self.joint_n == 22:
                joint_link_ = [(1, 2), (2, 3), (3, 4),
                               (5, 6), (6, 7), (7, 8),
                               (1, 9), (5, 9),
                               (9, 10), (10, 11), (11, 12),
                               (10, 13), (13, 14), (14, 15), (15, 16), (15, 17),
                               (10, 18), (18, 19), (19, 20), (20, 21), (20, 22)]
            if self.joint_n == 17:
                joint_link_ = [(1, 2), (2, 3),
                               (4, 5), (5, 6),
                               (1, 7), (4, 7),
                               (7, 8), (8, 9),
                               (8, 10), (10, 11), (11, 12), (11, 13),
                               (8, 14), (14, 15), (15, 16), (15, 17)]
            if self.joint_n == 11:
                joint_link_ = [(1,2),(3,4),(5,1),(5,3),(5,6),(6,7),(6,8),(8,9),(6,10),(10,11),(3,10),(1,8)]
            if self.joint_n == 9:
                joint_link_ = [(1,3),(2,3),(3,4),(4,5),(4,6),(6,7),(4,8),(8,9),(1,8),(2,6)]
            if self.joint_n == 6:
                joint_link_ = [(1,3),(2,3),(3,4),(3,5),(3,6),(1,6),(2,5)]
            if self.joint_n == 2:
                joint_link_ = [(1,2)]
        if skl_type == 'amass':
            if self.joint_n == 18:
                joint_link_ = [(1, 4), (4, 7), (2, 5), (5, 8),
                               (1, 3), (2, 3), (3, 6), (6, 9), (9, 12),
                               (9, 11), (11, 14), (14, 16), (16, 18),
                               (9, 10), (10, 13), (13, 15), (15, 17)]
        joint_link = [(i-1,j-1) for (i,j) in joint_link_]
        self.bone = self_link + joint_link

    def get_skeleton(self):
        skl = np.zeros((self.joint_n, self.joint_n))
        for i, j in self.bone:
            skl[j, i] = 1
            skl[i, j] = 1
        self.skeleton = skl