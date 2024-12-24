import sys
import os
import re
import time
import copy
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl.core import inline_functions
from ansys.mapdl import core as pymapdl
from ansys.mapdl import reader as pymapdl_reader
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from numpy.matlib import repmat
import eventlet
#https://mapdl.docs.pyansys.com/
#https://www.dandelioncloud.cn/article/details/1513086501389942785
eventlet.monkey_patch(select=True)
np.set_printoptions(threshold  = sys.maxsize)

class DG_Auto_selfmod():
    """Automatic modeling & Damage detection & Working condition analysis using ANSYS"""
    
    def __init__(self, path_pack, args):
        
        # ansys
        
        self.args = args
        # 骨架提取

        self.Using_point_ring = args.Using_point_ring# false for Oscar, ture for our implementation
        # PC_color = {1, .73, .0, 0.5f} # gold
        # PC_color = {0.275, .337, .60, 0.5f} # blue
        # self.PC_color = [1.0, .65, .35] # point cloud color, orange
        self.Neighbor_num_min = args.Neighbor_num_min
        self.Neighbor_num_max = args.Neighbor_num_max

        self.Laplacian_contraction_weight = args.Laplacian_contraction_weight # init Laplacian contraction weight(WL), compute automatically now
        self.Position_constraint_weight = args.Position_constraint_weight # init position constraint weight(WH)
        self.Laplacian_contraction_scale = args.Laplacian_contraction_scale # scalar for increasing WL in each iteration
        self.Laplacian_contraction_weight_max = args.Laplacian_contraction_weight_max # 2048
        self.Position_constraint_weight_max = args.Position_constraint_weight_max # 10000
        self.contraction_num_max = args.contraction_num_max # max contract iterations 20
        self.contraction_termination_condition = args.contraction_termination_condition # contract Termination Conditions for total area ratio 0.01

        self.radius = args.radius
        self.connect_r = args.connect_r
        self.branch_t = args.branch_t

        # self.Stress_t = args.Stress_t

############################################## 骨架提取 ##############################################

    def compute_k_knn(self, npts):
        
        num = np.max([self.Neighbor_num_min, np.around(npts*0.012).astype(int)])
        if num > self.Neighbor_num_max:
            num = self.Neighbor_num_max
        
        return num
    
    def normalize(self, pts, s_flag = True):
        # scale to unitBox and move to origin
        bbox = np.array([np.min(pts[:,0]),np.min(pts[:,1]),np.min(pts[:,2]),
                         np.max(pts[:,0]),np.max(pts[:,1]),np.max(pts[:,2])])
        if s_flag:
            self.c = (bbox[0:3]+bbox[3:6])*0.5
            pts = pts - self.c
            self.s = 1.6 / np.max(bbox[3:6] - bbox[0:3])
            pts = pts * self.s
        else:
            self.c = (bbox[0:3]+bbox[3:6])*0.5
            pts = pts - self.c

        return pts
    
    def compute_bbox(self, pts):
          
        bbox = np.array([np.min(pts[:,0]),np.min(pts[:,1]),np.min(pts[:,2]),
                         np.max(pts[:,0]),np.max(pts[:,1]),np.max(pts[:,2])])
        rs = bbox[3:6] - bbox[0:3]
        diameter = np.sqrt(np.dot(rs,rs.T))

        return bbox, diameter
    
    def compute_init_laplacian_constraint_weight(self, pts, rings):

        ms = self.one_ring_size(pts, rings, 2)
        wl = 1.0 / (5.0 * np.mean(ms))

        return wl
    
    def one_ring_size(self, pts, rings, type):
        
        n = pts.shape[0]
        ms = np.zeros((n,1))
        if type == 1:
            for i in range(n):
                ring = rings[i]
                tmp = repmat(pts[i,:], len(ring), 1) - pts[ring,:]
                ms[i] = np.min(np.sum(tmp**2, 1)**0.5)
        elif type == 2:
            for i in range(n):
                ring = rings[i]   
                tmp = repmat(pts[i,:], len(ring), 1) - pts[ring,:]
                ms[i] = np.mean(np.sum(tmp**2, 1)**0.5)                
        elif type == 3:
            for i in range(n):
                ring = rings[i]   
                tmp = repmat(pts[i,:], len(ring), 1) - pts[ring,:]
                ms[i] = np.max(np.sum(tmp**2, 1)**0.5)

        return ms
    
    def read_mesh(self, filename):
        # 'vertex' is a 'n x 3' array specifying the position of the vertices.
        vertex = 0

        return vertex
    
    def compute_point_point_ring(self, pts, k):
        # pts: n*3 matrix for coordinates where we want compute 1-ring.
        # k: k of kNN
        # index: index of kNN

        npts = pts.shape[0]
        ring = [[] for _ in range(npts)]

        # k近邻算法
        kdtree = KDTree(pts)
        # kNN
        dists, index = kdtree.query(pts, k)

        # print(index)

        for i in range(npts):
            neighbor = pts[index[i,:],:]
            # print(neighbor)
            coefs = (PCA().fit(neighbor).components_).T[:,:2]
            x = np.concatenate((neighbor.dot(coefs[:, 0]).reshape(-1,1), neighbor.dot(coefs[:, 1]).reshape(-1,1)), axis=1)

            # print(x)
            tri = (Delaunay(x)).simplices
            # print(tri)

            row, col = np.where(tri == 0)
            temp = copy.deepcopy(tri[row,:])

            temp = np.sort(temp, 1)
            temp = copy.deepcopy(temp[:, 1:])

            x = temp.reshape(-1,1)
            x = np.sort(x, 0)
            x_1 = np.row_stack((x, np.max(x,0)+1))
            d = np.diff(x_1, axis=0)
            d_1 = np.insert(d, 0, [1], axis=0)
            count = np.diff(np.where(d_1)[0],axis=0).reshape(-1,1)
            y = np.concatenate((x[np.where(d)].reshape(-1,1), count), axis=1)
            n_sorted_index = y.shape[0]
            start = np.where(count == 1)[0]

            if start.size:
                want_to_find = y[start[0], 0]
            else:
                want_to_find = temp[0, 0]
                n_sorted_index += 1
            
            sorted_index = np.zeros((1, n_sorted_index))
            for j in range(n_sorted_index):
                sorted_index[0,j] = want_to_find
                row, col = np.where(temp == want_to_find)
                if col.size:
                    if col[0] == 0:
                        want_to_find = temp[row[0],1]
                        temp[row[0],1] = -1
                    else:
                        want_to_find = temp[row[0],0]
                        temp[row[0],0] = -1
            neighbor_index = index[i, sorted_index.astype(int)]

            ring[i] = neighbor_index
        
        return ring
    
    def compute_point_laplacian(self, pts, rings):

        n = pts.shape[0]
        W = np.zeros((n,n))

        for i in range(n):
            ring = rings[i][0]

            tmp = len(ring) - 1
            for ii in range(tmp):
                j = ring[ii]
                k = ring[ii+1]
                vi = pts[i,:]
                vj = pts[j,:]
                vk = pts[k,:]

                # new % Oscar08 use this 
                u = vk-vi
                v = vk-vj
                cot1 = np.dot(u,v) / np.linalg.norm(np.cross(u,v))
                W[i,j] = W[i,j] + cot1
                u = vj-vi
                v = vj-vk
                cot2 = np.dot(u,v) / np.linalg.norm(np.cross(u,v))
                W[i,k] = W[i,k] + cot2

        L = np.diag(np.sum(W, 1)) - W

        return L
    
    def compute_cosine_laplacian(self, pts_cos, rings):

        n = pts_cos.shape[0]
        W = np.zeros((n,n))

        for i in range(n):
            ring = rings[i][0]

            tmp = len(ring)
            for ii in range(tmp):
                j = ring[ii]
                u = pts_cos[i,:]
                v = pts_cos[j,:]
                
                u_norm = np.linalg.norm(u)
                v_norm = np.linalg.norm(v)
                cos = np.dot(u,v)/(u_norm * v_norm)
                # tan1 = np.linalg.norm(np.cross(u,v)) / np.dot(u,v)
                c = np.exp(cos)

                W[i,j] = W[i,j] + c
        
        L = np.diag(np.sum(W, 1)) - W

        return L
    
    def compute_weight_laplacian(self, pts_weight, rings):

        n = pts_weight.shape[0]
        W = np.zeros((n,n))

        for i in range(n):
            ring = rings[i][0]

            tmp = len(ring)
            wmin = np.min(pts_weight[ring])
            wmax = np.max(pts_weight[ring])
            for ii in range(tmp):
                j = ring[ii]
                wj = pts_weight[j]
                w = np.exp((wj-wmin)/(wmax-wmin))

                W[i,j] = W[i,j] + w
        
        L = np.diag(np.sum(W, 1)) - W

        return L

    def compute_cosineweight_laplacian(self, pts_cos, pts_weight, rings):

        n = pts_weight.shape[0]
        W = np.zeros((n,n))

        for i in range(n):
            ring = rings[i][0]

            tmp = len(ring)
            wmin = np.min(pts_weight[ring])
            wmax = np.max(pts_weight[ring])
            for ii in range(tmp):
                j = ring[ii]
                wj = pts_weight[j]
                w = np.exp((wj-wmin)/(wmax-wmin))

                u = pts_cos[i,:]
                v = pts_cos[j,:]

                u_norm = np.linalg.norm(u)
                v_norm = np.linalg.norm(v)
                cos = np.dot(u,v)/(u_norm * v_norm)
                # tan1 = np.linalg.norm(np.cross(u,v)) / np.dot(u,v)
                c = np.exp(1-cos)

                W[i,j] = W[i,j] + c * w
        
        L = np.diag(np.sum(W, 1)) - W

        return L

    def contraction_by_mesh_laplacian(self, pts, npts, k_knn, rings, bbox, pts_cos, pts_weight):
        
        # Laplace_type = 'conformal'

        ring_size_type = 1 # 1: min, 2:mean, 3:max
        show_contraction_progress = True

        tc = self.contraction_termination_condition
        iterate_time = self.contraction_num_max
        iterate_time = 1

        initWL = self.compute_init_laplacian_constraint_weight(pts, rings)
        WC = 1
        WH = np.ones((npts)) * WC # 初始约束权 initial attraction weight
        sl = self.Laplacian_contraction_scale # 3
        # WL = initWL # 初始收缩权 initial contraction weight
        WL = 20

        # init iteration
        t = 1 # current iteration step

        # left side of the equation
        L = 0
        for i in range(6):
            L += (- self.compute_cosineweight_laplacian(pts_cos[:,3*i:3*(i+1)], pts_weight, rings))/6

        # L = 0
        # for i in range(6):
        #     L += (- self.compute_point_laplacian(pts_cos[:,3*i:3*(i+1)], pts_weight, pts, rings))/6

        # L = 0
        # for i in range(6):
        #     L += (- self.compute_cosine_laplacian(pts_cos[:,3*i:3*(i+1)], rings))/6
        
        # L = - self.compute_weight_laplacian(pts_weight, rings)
        

        A = np.concatenate((L*WL, np.diag(WH)), axis = 0)

        # right side of the equation
        b = np.concatenate((np.zeros((npts,3)), np.diag(WH).dot(pts)), axis = 0)

        # solve
        cpts = np.dot(np.linalg.pinv((A.T).dot(A)), (A.T).dot(b))

        self.scatter_ploter(cpts, pts, 2)
        # self.scatter_ploter(cpts, pts)

        sizes = self.one_ring_size(pts, rings, ring_size_type)  # min radius of 1-ring
        size_new = self.one_ring_size(cpts, rings, ring_size_type)
        a_1 = np.sum(size_new)/np.sum(sizes)
        # print(a_1)

        while t < iterate_time:
            L = - self.compute_point_laplacian(cpts, rings) # conformal

            WL = sl * initWL

            if WL > self.Laplacian_contraction_weight_max:
                WL = self.Laplacian_contraction_weight_max
            
            WH = (sizes/size_new) * WC

            WH[WH > self.Position_constraint_weight_max] = self.Position_constraint_weight_max

            # update left side of the equation
            A = np.real(np.concatenate((WL*L, np.diag(WH.reshape(1,-1)[0])), axis = 0))

            # update right side of the equation
            b[npts:, :] = np.diag(WH.reshape(1,-1)[0]).dot(cpts)

            # solve
            tmp = np.dot(np.linalg.pinv((A.T).dot(A)), (A.T).dot(b))

            self.scatter_ploter(tmp, cpts, 2)

            size_new = self.one_ring_size(tmp, rings, ring_size_type) 
            a_2 = np.sum(size_new)/np.sum(sizes)

            tmpbox = self.compute_bbox(tmp)[0]

            # if np.sum( (tmpbox[3:] - tmpbox[:3]) > ((bbox[3:] - bbox[:3])*1.2) ) > 0:
            #     break

            if a_1 - a_2 < tc:
                cpts = tmp
                print(t, 'Iteration')
                print('Convergence:', a_1 - a_2)
                break
            else:
                cpts = tmp
                a_1 = a_2
            
            print(t, 'Iteration')
            t += 1
        
        return (cpts, t, initWL, WC, sl)
    
    def farthest_sampling_by_sphere(self, pts, RADIUS):
        # FURTHEST POINT DOWNSAMPLE THE CLOUD

        kdtree = KDTree(pts)
        spls = np.zeros((1,3))
        corresp = np.ones((pts.shape[0],1)) * -1

        for k in range(pts.shape[0]):
            if corresp[k,0] != -1:
                continue
            
            # query all the points for distances

            # initialize the priority queue
            maxIdx = k

            # factor = 1
            # # query its delta-neighborhood
            # while True:
                # nIdxs_0, nDsts_0 = kdtree.query_radius([pts[maxIdx,:]], RADIUS * factor, return_distance=True)

                # if len(nIdxs_0[0]) < 2:
                #     break

                # C = ((pts[nIdxs_0[0],:] - np.mean(pts[nIdxs_0[0],:], axis=0)).T).dot((pts[nIdxs_0[0],:] - np.mean(pts[nIdxs_0[0],:], axis=0)))
                # U_sing,Sigema_sing,V_sing = np.linalg.svd(C, full_matrices = 1, compute_uv = 1)

                # l_2 = Sigema_sing[0]
                # l_1 = Sigema_sing[1]
                # l_0 = Sigema_sing[2]

                # lv = l_2/(l_0 + l_1 + l_2)

                # if lv > 0.9:
                #     break
                # else:
                #     factor = factor * 0.5
            nIdxs_0, nDsts_0 = kdtree.query_radius([pts[maxIdx,:]], RADIUS, return_distance=True)

            nIdxs = nIdxs_0[0]
            nDsts = nDsts_0[0]

            # if maxIdx and all its neighborhood has been marked, skip ahead
            if np.all(corresp[nIdxs,0] != -1):
                continue

            # create new node and update (closest) distances
            spls = np.r_[spls, [pts[maxIdx,:]]] # ok<AGROW>

            corresp[nIdxs,0] = spls.shape[0] - 2
            
        spls = np.delete(spls, 0, axis = 0)

        corresp = corresp.astype(int)

        return spls, corresp
    
    def connect_by_inherit_neigh(self, spls, RADIUS):
        # build a connection matrix of downsamples (spls) by inherit neighbors of samples (pts) they correspond

        n_s = spls.shape[0]

        # k_knn = self.compute_k_knn(n_s)
        k_knn = 6

        kdtree = KDTree(spls)
        # kNN
        dists, index = kdtree.query(spls, k_knn)

        ind, dset = kdtree.query_radius(spls, r = RADIUS * self.connect_r, return_distance=True)

        A = np.zeros((n_s,n_s))

        for pIdx in range(n_s):
            # 数量邻域连接
            ns = index[pIdx,1:]
            # i = 1
            for nIdx in ns:
                # if dists[pIdx,i] <= 3 * RADIUS:
                    A[pIdx,nIdx] = 1
                    A[nIdx,pIdx] = 1
                # i += 1
                # break # new method, generates less edges
                # continue # old method, generates more edges

            # # 半径邻域连接
            # ns_1 = ind[pIdx]
            # for nIdx in ns_1:
            #     A[pIdx,nIdx] = 1
            #     A[nIdx,pIdx] = 1
            #     # break # new method, generates less edges
            #     continue # old method, generates more edges

        A[np.where(A > 0)] = 1

        A = A - np.identity(n_s)

        A[np.where(A < 0)] = 0

        return A
    
    def euclidean_distance(self, p1, p2):
        v = p1 - p2
        dist = np.sqrt(np.dot(v,v.T))

        return dist
    
    def edge_collapse_update(self, spls, spls_adj, corresp):

        n_s = spls.shape[0]
        collapse_order = 0
        
        # recover the set of edges on triangles & count triangles
        A = copy.deepcopy(spls_adj)

        degrees = np.zeros((n_s,1))
        for i in range(n_s):
            ns = np.where(A[i,:] == 1)[0]
            degrees[i,0] = len(ns)

        tricount = 0
        skeds = np.zeros((1,4)) # idx1, idx2, average degree of two end points, distance

        for i in range(n_s):
            ns = np.where(A[i,:] == 1)[0]
            ns_1 = ns[np.where(ns > i)] # touch every triangle only once (if a edge belong to two triangles, it appears twice!)
            lns = len(ns_1)
            if lns >= 2:
                for j in range(lns):
                    for k in range(j+1,lns):
                        if A[ns_1[j],ns_1[k]] == 1:
                            tricount += 1

                            # print(i,ns_1[j],ns_1[k])

                            skeds = np.r_[skeds, [[i, ns_1[j], 0.5*(lns + degrees[ns_1[j],0]), 0]]]
                            skeds[-1,3] = self.euclidean_distance(spls[i,:], spls[ns_1[j],:])
                            # print(skeds[-1,3])

                            skeds = np.r_[skeds, [[ns_1[j], ns_1[k], 0.5*(degrees[ns_1[j],0] + degrees[ns_1[k],0]), 0]]]
                            skeds[-1,3] = self.euclidean_distance(spls[ns_1[j],:], spls[ns_1[k],:])
                            # print(skeds[-1,3])

                            skeds = np.r_[skeds, [[i, ns_1[k], 0.5*(lns + degrees[ns_1[k],0]), 0]]]
                            skeds[-1,3] = self.euclidean_distance(spls[ns_1[k],:], spls[i,:])
                            # print(skeds[-1,3])
        
        # print(tricount)
        skeds = np.delete(skeds, 0, axis = 0)
        
        # EDGE COLLAPSE 
        while True:

            # STOP CONDITION
            # no more triangles? then the structure is 1D
            if skeds.shape[0] == 0:
                break
            
            # DECIMATION STEP + UPDATES
            # collapse the edge with minimum cost, remove the second vertex
            if collapse_order == 1: # cost is degree + distance
                mind = np.min(skeds[:,2])
                tmpIdx = np.where(skeds[:,2] == mind)[0]
                tmpSkeds = skeds[tmpIdx,3]

                idx = np.argmin(tmpSkeds)
                edge = skeds[tmpIdx[idx], :2]
                skeds = np.delete(skeds, tmpIdx[idx], axis = 0)
            else: # cost is distance
                idx = np.argmin(skeds[:,3])
                edge = skeds[idx, :2]
                skeds= np.delete(skeds, idx, axis = 0)
            
            edge = edge.astype(int)
            # print(edge)
            # update the location
            spls[edge[1],:] = np.mean(spls[edge[:],:], axis=0)
            spls[edge[0],:] = -10

            # update the A matrix
            A[edge[1],:] += A[edge[0],:]
            A[:,edge[1]] += A[:,edge[0]]

            A[edge[0],:] = 0
            A[:,edge[0]] = 0
            
            A[np.where(A > 0)] = 1

            # update the correspondents
            corresp[np.where(corresp == edge[0])] = edge[1]
                
            # update distance and degree of edges
            degrees = np.zeros((n_s,1))
            for i in range(n_s):
                ns = np.where(A[i,:] == 1)[0]
                degrees[i,0] = len(ns)

            tricount = 0
            skeds = np.zeros((1,4)) # idx1, idx2, average degree of two end points, distance

            for i in range(n_s):
                ns = np.where(A[i,:] == 1)[0]
                ns_1 = ns[np.where(ns > i)] # touch every triangle only once (if a edge belong to two triangles, it appears twice!)
                lns = len(ns_1)
                if lns >= 2:
                    for j in range(lns):
                        for k in range(j+1,lns):
                            if A[ns_1[j],ns_1[k]] == 1:
                                tricount += 1

                                # print(i,ns_1[j],ns_1[k])

                                skeds = np.r_[skeds, [[i, ns_1[j], 0.5*(lns + degrees[ns_1[j],0]), 0]]]
                                skeds[-1,3] = self.euclidean_distance(spls[i,:], spls[ns_1[j],:])
                                # print(skeds[-1,3])

                                skeds = np.r_[skeds, [[ns_1[j], ns_1[k], 0.5*(degrees[ns_1[j],0] + degrees[ns_1[k],0]), 0]]]
                                skeds[-1,3] = self.euclidean_distance(spls[ns_1[j],:], spls[ns_1[k],:])
                                # print(skeds[-1,3])

                                skeds = np.r_[skeds, [[i, ns_1[k], 0.5*(lns + degrees[ns_1[k],0]), 0]]]
                                skeds[-1,3] = self.euclidean_distance(spls[ns_1[k],:], spls[i,:])
                                # print(skeds[-1,3])
            
            # print(tricount)
            skeds = np.delete(skeds, 0, axis = 0)
        
        A = np.delete(A, np.where(spls[:,0] == -10)[0], axis = 0)
        A = np.delete(A, np.where(spls[:,0] == -10)[0], axis = 1)
        
        spls = np.delete(spls, np.where(spls[:,0] == -10)[0], axis = 0)

        A[np.where(A > 0)] = 1

        A = A - np.identity(spls.shape[0])

        A[np.where(A < 0)] = 0
        
        return spls, A, corresp
    
    def rosa_lineextract(self, cpts, sample_radius):
        
        spls_0, corresp = self.farthest_sampling_by_sphere(cpts, sample_radius)
        spls_connect = self.connect_by_inherit_neigh(spls_0, sample_radius)
        self.plot_result(cpts, spls_0, spls_connect)
        spls_1 = copy.deepcopy(spls_0)
        spls_connect_1 = copy.deepcopy(spls_connect)
        spls, spls_adj, corresp = self.edge_collapse_update(spls_1, spls_connect_1, corresp)

        return spls_0, spls_connect, spls, spls_adj, corresp
    
    def cal_degrees(self, spls, spls_adj):

        n_s = spls.shape[0]
        A = copy.deepcopy(spls_adj)

        degrees = np.zeros((n_s,1))
        for i in range(n_s):
            ns = np.where(A[i,:] == 1)[0]
            degrees[i,0] = len(ns)
        
        return degrees
    
    def regularization(self, spls_0, spls_adj_0, jiao_t):
        # 正则化, 变曲为直

        spls = copy.deepcopy(spls_0)
        spls_adj = copy.deepcopy(spls_adj_0)
        
        degrees = self.cal_degrees(spls, spls_adj)

        InterP = np.where(degrees[:,0] == 2)[0]

        for i in InterP:
            j = np.where(spls_adj[i,:] == 1)[0][0]
            k = np.where(spls_adj[i,:] == 1)[0][1]
            
            u = spls[j,:] - spls[i,:]
            v = spls[k,:] - spls[i,:]

            u_norm = np.linalg.norm(u)
            v_norm = np.linalg.norm(v)
            cos = np.dot(u,v)/(u_norm * v_norm)

            if np.abs(cos) >= np.cos(np.pi/180*jiao_t):
                spls_adj[j,k] = 1
                spls_adj[k,j] = 1
                spls_adj[i,:] = 0
                spls_adj[:,i] = 0
                spls[i,:] = -10
        
        spls_adj = np.delete(spls_adj, np.where(spls[:,0] == -10)[0], axis = 0)
        spls_adj = np.delete(spls_adj, np.where(spls[:,0] == -10)[0], axis = 1)
        
        spls = np.delete(spls, np.where(spls[:,0] == -10)[0], axis = 0)

        spls_adj[np.where(spls_adj > 0)] = 1

        spls_adj = spls_adj - np.identity(spls.shape[0])

        spls_adj[np.where(spls_adj < 0)] = 0
        
        return spls, spls_adj
    
    def cal_branch(self, spls, spls_adj):
        # 查找分支及计算长度

        A = copy.deepcopy(spls_adj)

        degrees = self.cal_degrees(spls, spls_adj)
        
        DP = np.where(degrees[:,0] == 1)[0]

        length_list = np.zeros((len(DP),1))
        list_branch = []
        repeatD = []
        index = 0

        for i in DP:
            if i in repeatD:
                continue

            A_temp = copy.deepcopy(A)
            lastP = i
            branch_list = []
            branch_list.append(lastP)
            while True:
                nextP = np.where(A_temp[lastP,:] == 1)[0]

                if len(nextP) > 1:
                    branch_list.pop(-1)
                    cos = 0
                    for j in nextP:
                        
                        u = spls[j,:] - spls[lastP,:]
                        v = spls[i,:] - spls[lastP,:]

                        u_norm = np.linalg.norm(u)
                        v_norm = np.linalg.norm(v)
                        cos += np.abs(np.dot(u,v)/(u_norm * v_norm))

                    cos = cos/len(nextP)
                    length_list[index,0] = length_list[index,0]*(cos**4)

                    break
                elif len(nextP) == 0:
                    repeatD.append(lastP)
                    break
                else:
                    nextP = nextP[0]
                    branch_list.append(nextP)
                    A_temp[nextP, lastP] = 0
                    length_list[index,0] += self.euclidean_distance(spls[lastP,:], spls[nextP,:])
                    lastP = nextP
            
            index += 1
            list_branch.append(branch_list)
        
        length_list = np.delete(length_list, np.where(length_list[:,0] == 0)[0], axis = 0)

        return DP, length_list, list_branch
    
    def remove_spls(self, spls, spls_adj, list_remove):

        A = copy.deepcopy(spls_adj)
        spls = np.delete(spls, list_remove, axis = 0)
        A = np.delete(A, list_remove, axis = 0)
        A = np.delete(A, list_remove, axis = 1)

        return spls, A
    
    def regularization_xy(self, spls_0, spls_adj_0, xy_t):
        # 正则化, 规整平面
        spls = copy.deepcopy(spls_0)
        spls_adj = copy.deepcopy(spls_adj_0)

        x = [1,0,0]
        y = [0,1,0]

    def DG_contraction(self, pts_0):

        # Stress_pts: x, y, z, G-S1, G-S3, G-V1x, G-V1y, G-V1z, G-V3x, G-V3y, G-V3z
        #                    , O-S1, O-S3, O-V1x, O-V1y, O-V1z, O-V3x, O-V3y, O-V3z
        #                    , I-S1, I-S3, I-V1x, I-V1y, I-V1z, I-V3x, I-V3y, I-V3z, varray
        # varray: the number of the component to which the point belongs
          
        # pts_0 = self.read_file(filename)

        print('######### Skeletons Contraction #########')

        self.color1 = '#FF6347' # 红
        self.color2 = '#9ACD32' # 绿
        self.color3 = '#00BFFF' # 蓝

        # self.Stress_t = 50
            
        # print('Stress Points Percent:', 100 - self.Stress_t, '%')
        pts_weight = np.concatenate((pts_0[:,3:5], pts_0[:,11:13], pts_0[:,19:21]), axis = 1)
        # print(pts_weight.shape)
        pts_weight_1 = np.sum(pts_weight,1)

        pts_0 = pts_0[np.where(pts_weight_1 != 0)[0],:]
        pts_ini = copy.deepcopy(pts_0)

        pts_0[:,:3] = self.normalize(pts_0[:,:3], False)
        c_1 = self.c

        # 切半
        pts_0 = pts_0[np.where(pts_0[:,0] >= 0)[0],:]

        pts_weight = np.concatenate((pts_0[:,3:5], pts_0[:,11:13], pts_0[:,19:21]), axis = 1)
        pts_weight_1 = np.sum(pts_weight,1)

        pts_cos = np.concatenate((pts_0[:,5:11], pts_0[:,13:19], pts_0[:,21:27]), axis = 1)
        # print(pts_cos.shape)

        # S_W_P = np.percentile(pts_weight_1[np.where(pts_0[:,27] == 1)], self.Stress_t) # 让收敛点百分比迭代下降
        # pts_temp = pts_0[np.where(pts_0[:,27] == 1),:][0]
        # pts_cos_temp = pts_cos[np.where(pts_0[:,27] == 1),:][0]
        # weight_temp = pts_weight_1[np.where(pts_0[:,27] == 1)]
        # pts_1 = pts_temp[np.where(weight_temp > S_W_P)[0],:3]
        # pts_cos_1 = pts_cos_temp[np.where(weight_temp > S_W_P)[0],:]
        # pts_weight_2 = weight_temp[np.where(weight_temp > S_W_P)[0]]

        # for i in range(2,self.Volunum_1+1):
        #     if i == 3:
        #         continue
        #     S_W_P = np.percentile(pts_weight_1[np.where(pts_0[:,27] == i)], self.Stress_t) # 让收敛点百分比迭代下降
        #     pts_temp = pts_0[np.where(pts_0[:,27] == i),:][0]
        #     pts_cos_temp = pts_cos[np.where(pts_0[:,27] == i),:][0]
        #     weight_temp = pts_weight_1[np.where(pts_0[:,27] == i)]
        #     pts_1 = np.concatenate((pts_1, pts_temp[np.where(weight_temp > S_W_P)[0],:3]), axis = 0)
        #     pts_cos_1 = np.concatenate((pts_cos_1, pts_cos_temp[np.where(weight_temp > S_W_P)[0],:]), axis = 0)
        #     pts_weight_2 = np.concatenate((pts_weight_2, weight_temp[np.where(weight_temp > S_W_P)[0]]), axis = 0)

        # pts_1 = pts_0[np.where(pts_0[:,27] == 4),:3][0]
        # pts_cos_1 = copy.deepcopy(pts_cos[np.where(pts_0[:,27] == 4),:][0])
        # pts_weight_2 = copy.deepcopy(pts_weight_1[np.where(pts_0[:,27] == 4)])

        pts_1 = pts_0[:,:3]
        pts_cos_1 = copy.deepcopy(pts_cos)
        pts_weight_2 = copy.deepcopy(pts_weight_1)

        npts = pts_1.shape[0]
        # print(pts_1.shape)
        # print(pts_cos_1.shape)
        # print(pts_weight_2.shape)

        pts_norm = self.normalize(pts_1)
        s = self.s
        pts_norm_1 = self.normalize(pts_0[:,:3], False)
        c_2 = self.c
        
        bbox, diameter = self.compute_bbox(pts_norm)

        # Step 0: build local 1-ring
        k_knn = self.compute_k_knn(npts)
        # k_knn = 5

        rings = self.compute_point_point_ring(pts_norm, k_knn)
        # print(rings)

        # Step 1: Contract point cloud by Laplacian
        cpts, t, initWL, WC, sl = self.contraction_by_mesh_laplacian(pts_norm, npts, k_knn, rings, bbox, pts_cos_1, pts_weight_2)

        # self.scatter_ploter(cpts, pts_norm, 2)

        sample_radius = diameter * self.radius
        # spls_0, corresp_0 = self.farthest_sampling_by_sphere(cpts, sample_radius)

        # spls_connect = self.connect_by_inherit_neigh(spls_0, sample_radius)
        
        spls_0, spls_connect, spls, spls_adj, corresp = self.rosa_lineextract(cpts, sample_radius)
        
        self.plot_result(pts_norm, spls, spls_adj,flag_num = 1)

        spls_1, spls_adj_1 = self.regularization(spls, spls_adj, 30)

        spls_2, spls_adj_2 = self.regularization(spls_1, spls_adj_1, 60)

        degrees = self.cal_degrees(spls_2, spls_adj_2)

        # 修剪短分支
        if len(np.where(degrees[:,0] == 1)[0]): # 有枝条
            while True:
                DP, length_list, list_branch = self.cal_branch(spls_2, spls_adj_2)

                print(length_list)
                # print(list_branch)

                if length_list.shape[0] == 0:
                    break
                
                if np.min(length_list) > self.branch_t:
                    break

                toremove = list_branch[np.argmin(length_list)]

                spls_2, spls_adj_2 = self.remove_spls(spls_2, spls_adj_2, toremove)
            
        spls_3, spls_adj_3 = self.regularization(spls_2, spls_adj_2, 30)
        spls_4 = copy.deepcopy(spls_3)

        # 平面化
        spls_4[np.where(spls_4[:,0] < np.mean(spls_4[:,0]))[0],0] = np.min(spls_4[:,0])

        x_axis = np.min(spls_4[:,0])

        x = spls_4[:,0]
        y = spls_4[:,1]
        y_0 = y[np.where(x > np.min(x))[0]]
        y[np.where(np.abs(y - np.min(y_0)) <= (np.max(y_0) - np.min(y_0))/5)[0]] = np.min(y_0)
        y[np.where(np.abs(y - (np.max(y_0)+np.min(y_0))/2) <= (np.max(y_0) - np.min(y_0))/5)[0]] = (np.max(y_0)+np.min(y_0))/2
        y[np.where(np.abs(y - np.max(y_0)) <= (np.max(y_0) - np.min(y_0))/5)[0]] = np.max(y_0)

        spls_4[:,1] = y
        spls_adj_4 = copy.deepcopy(spls_adj_3)

        # 去除重合点
        kdtree_1 = KDTree(spls_4)
        # kNN
        dists, index = kdtree_1.query(spls_4, 2)
        repeatD = []

        for i in np.where(dists[:,1] < 0.1)[0]:
            if i in repeatD:
                continue

            toremove = index[i,1]
            tokeep = index[i,0]
            if spls_4[toremove,0] > np.min(x) or spls_4[tokeep,0] > np.min(x):
                continue

            repeatD.append(toremove)

            spls_4[tokeep,:] = (spls_4[tokeep,:] + spls_4[toremove,:])/2
            
            spls_adj_4[tokeep,:] += spls_adj_4[toremove,:]
            spls_adj_4[:,tokeep] += spls_adj_4[:,toremove]

        spls_4 = np.delete(spls_4, repeatD, axis = 0)
        spls_adj_4 = np.delete(spls_adj_4, repeatD, axis = 0)
        spls_adj_4 = np.delete(spls_adj_4, repeatD, axis = 1)

        spls_adj_4[np.where(spls_adj_4 > 0)] = 1

        # 镜像
        p_temp = spls_4[np.where(spls_4[:,0] > np.min(x))[0]]

        spls_5 = np.zeros((spls_adj_4.shape[0] + p_temp.shape[0],p_temp.shape[1]))
        spls_5[:spls_adj_4.shape[0],:] = spls_4
        spls_5[spls_adj_4.shape[0]:,:] = p_temp
        spls_5[spls_adj_4.shape[0]:,0] = 2 * np.min(x) - spls_5[spls_adj_4.shape[0]:,0]

        spls_adj_5 = np.zeros((spls_adj_4.shape[0] + p_temp.shape[0], spls_adj_4.shape[0] + p_temp.shape[0]))
        spls_adj_5[:spls_adj_4.shape[0],:spls_adj_4.shape[0]] = spls_adj_4[:,:]
        temp_1 = spls_adj_4[np.where(spls_4[:,0] <= np.min(x))[0], :]
        temp_2 = temp_1[:, np.where(spls_4[:,0] > np.min(x))[0]]
        spls_adj_5[np.where(spls_4[:,0] <= np.min(x))[0],spls_adj_4.shape[0]:] = temp_2
        temp_3 = spls_adj_4[np.where(spls_4[:,0] > np.min(x))[0], :]
        temp_4 = temp_3[:, np.where(spls_4[:,0] <= np.min(x))[0]]
        spls_adj_5[spls_adj_4.shape[0]:,np.where(spls_4[:,0] <= np.min(x))[0]] = temp_4
        temp_5 = spls_adj_4[np.where(spls_4[:,0] > np.min(x))[0], :]
        temp_6 = temp_5[:, np.where(spls_4[:,0] > np.min(x))[0]]
        spls_adj_5[spls_adj_4.shape[0]:,spls_adj_4.shape[0]:] = temp_6
        spls_5[:,0] = spls_5[:,0] - x_axis
        
        pts_fa = spls_4 / s
        pts_fa_2 = spls_5 / s + c_1
        # pts_fa = spls / self.s + self.c
        
        self.plot_result(pts_norm_1, pts_fa, spls_adj_4,flag_num = 1)
        self.plot_result(pts_ini, pts_fa_2, spls_adj_5,flag_num = 1)
        self.plot_result(pts_ini, pts_fa_2, spls_adj_5)
        self.scatter_ploter(pts_ini, cpts, 1)
        self.scatter_ploter(pts_norm_1, cpts, 1)
        # self.plot_result_2(pts_0[:,:3], pts_fa, spls_adj, pts_norm / self.s + self.c)

        return pts_ini, pts_norm, cpts, spls_0, spls_connect, spls, spls_adj, spls_4, spls_adj_4, spls_5, spls_adj_5, pts_fa_2
        # return pts_norm, cpts, t
    
    def scatter_ploter(self, cpts, pts_norm = 0, num = 1, a = 10, b = 20):
        
        label_font = { 'color': 'c', 'size': 15, 'weight': 'bold'}

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection="3d")  # 添加子坐标轴，111表示1行1列的第一个子图

        x = cpts[:,0]
        y = cpts[:,1]
        z = cpts[:,2]
        ax.scatter(x, y, z, c = self.color1, marker = '*', label= 'cpts', s = 100) #这里marker的尺寸和z的大小成正比  
              
        if num == 2:
            x = pts_norm[:,0]
            y = pts_norm[:,1]
            z = pts_norm[:,2]
            ax.scatter(x, y, z, c = self.color2, marker = 'o', label= 'pts_norm', s = 25, alpha = 0.5) #这里marker的尺寸和z的大小成正比

        ax.set_xlabel("X axis", fontdict=label_font)
        ax.set_ylabel("Y axis", fontdict=label_font)
        ax.set_zlabel("Z axis", fontdict=label_font)
        ax.set_title("Scatter plot", alpha=0.6, color="b", size=25, weight='bold', backgroundcolor="y")   #子图的title
        ax.legend(loc="upper left")    #legend的位置左上

        ax.view_init(a, b)

        plt.show()
    
    def plot_connectivity(self, pts, A, pts_1 = 0, A_1 = 0, count = 1):
           
        fig = plt.figure(figsize=(16, 12))
        ay = fig.add_subplot(111, projection="3d")
        label_font = { 'color': 'c', 'size': 15, 'weight': 'bold'}
        
        edge_num_0 = 0

        for i in range(A.shape[0]-1):
            for j in range(i+1, A.shape[1]):
                if(A[i,j]>0):
                    idx = np.array([[i],[j]])
                    ay.plot(pts[idx,0], pts[idx,1], pts[idx,2], c = self.color1, linewidth = 5)
                    # text(pts(i,1),pts(i,2),pts(i,3),int2str(i))
                    edge_num_0 += 1
        
        if count == 2:
            edge_num_1 = 0

            for i in range(A_1.shape[0]-1):
                for j in range(i+1, A_1.shape[1]):
                    if(A_1[i,j] > 0):
                        idx = np.array([[i],[j]])
                        ay.plot(pts_1[idx,0], pts_1[idx,1], pts_1[idx,2], c = self.color2, linewidth = 5, alpha = 0.5)
                        # text(pts(i,1),pts(i,2),pts(i,3),int2str(i))
                        edge_num_1 += 1

        
        ay.set_xlabel("X axis", fontdict=label_font)
        ay.set_ylabel("Y axis", fontdict=label_font)
        ay.set_zlabel("Z axis", fontdict=label_font)
        ay.set_title("Line plot", alpha=0.6, color="b", size=25, weight='bold', backgroundcolor="y")   #子图的title

        # ay.view_init(90, -90)

        plt.show()
    
    def plot_result(self, pts_norm, pts, A, a = 10, b = 20, flag_num = 2):

        fig = plt.figure(figsize=(16, 12))
        ay = fig.add_subplot(111, projection="3d")
        label_font = { 'color': 'c', 'size': 15, 'weight': 'bold'}

        for i in range(A.shape[0]-1):
            for j in range(i+1, A.shape[1]):
                if(A[i,j]>0):
                    idx = np.array([[i],[j]])
                    ay.plot(pts[idx,0], pts[idx,1], pts[idx,2], c = self.color3, linewidth = 4)
        
        if flag_num == 2:
            x_0 = pts_norm[:,0]
            y_0 = pts_norm[:,1]
            z_0 = pts_norm[:,2]
            ay.scatter(x_0, y_0, z_0, c = self.color2, marker = 'o', label= 'pts_norm', s = 25, alpha = 0.5) #这里marker的尺寸和z的大小成正比

        x_1 = pts[:,0]
        y_1 = pts[:,1]
        z_1 = pts[:,2]
        ay.scatter(x_1, y_1, z_1, c = self.color1, marker = '*', label= 'pts_fa', s = 150, alpha = 0.5) #这里marker的尺寸和z的大小成正比

        
        ay.set_xlabel("X axis", fontdict=label_font)
        ay.set_ylabel("Y axis", fontdict=label_font)
        ay.set_zlabel("Z axis", fontdict=label_font)
        ay.set_title("Result plot", alpha=0.6, color="b", size=25, weight='bold', backgroundcolor="y")   #子图的title
        ay.legend(loc="upper left")    #legend的位置左上

        ay.view_init(a, b)

        plt.show()
    
    def plot_result_2(self, pts_norm, pts, A, pts_1):

        fig = plt.figure(figsize=(16, 12))
        ay = fig.add_subplot(111, projection="3d")
        label_font = { 'color': 'c', 'size': 15, 'weight': 'bold'}

        for i in range(A.shape[0]-1):
            for j in range(i+1, A.shape[1]):
                if(A[i,j]>0):
                    idx = np.array([[i],[j]])
                    ay.plot(pts[idx,0], pts[idx,1], pts[idx,2], c = self.color3, linewidth = 4)
        
        x_0 = pts_norm[:,0]
        y_0 = pts_norm[:,1]
        z_0 = pts_norm[:,2]
        ay.scatter(x_0, y_0, z_0, c = self.color2, marker = 'o', label= 'pts_norm', s = 25, alpha = 0.5) #这里marker的尺寸和z的大小成正比

        x_1 = pts[:,0]
        y_1 = pts[:,1]
        z_1 = pts[:,2]
        ay.scatter(x_1, y_1, z_1, c = self.color1, marker = '*', label= 'pts_fa', s = 150, alpha = 0.5) #这里marker的尺寸和z的大小成正比

        x_2 = pts_1[:,0]
        y_2 = pts_1[:,1]
        z_2 = pts_1[:,2]
        ay.scatter(x_2, y_2, z_2, c = self.color1, marker = 'o', label= 'pts_1', s = 25, alpha = 0.5) #这里marker的尺寸和z的大小成正比

        
        ay.set_xlabel("X axis", fontdict=label_font)
        ay.set_ylabel("Y axis", fontdict=label_font)
        ay.set_zlabel("Z axis", fontdict=label_font)
        ay.set_title("Result plot", alpha=0.6, color="b", size=25, weight='bold', backgroundcolor="y")   #子图的title
        ay.legend(loc="upper left")    #legend的位置左上

        ay.view_init(0, 90)

        plt.show()