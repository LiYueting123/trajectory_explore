import sklearn.cluster as skc
import numpy as np
from sklearn import preprocessing
import pandas as pd
import csv
from math import *
import sys
sys.setrecursionlimit(10000000)
def Get_trajectories():
    trajectory_data_url = 'static/data/trajectories.txt'
    phonetraj_data_url='static/data/BS_WZ.csv'
    trajectories = []
    csvfile=open(phonetraj_data_url)
    csv_reader=csv.reader(csvfile)
    phonetraj={}
    for item in csv_reader:
        if csv_reader.line_num==1:
            continue
        if(item[4] !='' and (item[5] !='')):
            phonetraj[(int(item[1]),int(item[2]))] = (float(item[4]),float(item[5]))
    with open(trajectory_data_url, 'rt') as f:
        for line in f:
            # data:460005731942647;2014-01-14 06:50:05.91,26616 20288;2014-01-14 06:50:07.08,26616 20288;2014-01-14 06:55:13.27,26617 33844;2014-01-14 06:55:14.38,26617 33844;
            trajectory_inf = line.split(';')
            if (len(trajectory_inf)-2 < 10):
                continue
            trajectory = {}
            trajectory['id'] = trajectory_inf[0]
            trajectory_points = []  # 用于保存trajectory的各个点的位置和时间
            is_exist = True
            for i in range(1, len(trajectory_inf) - 1):  # 将每一个点保存
                point_inf = trajectory_inf[i].split(',')
                trajectory_point = {}
                trajectory_point['time'] = point_inf[0]
                location = point_inf[1].split(' ')
                if (location[0] !='') and (location[1]!='') and phonetraj.__contains__((int(location[0]),int(location[1]))):
                    (trajectory_point['x'],trajectory_point['y']) = phonetraj[(int(location[0]),int(location[1]))]# 如果有数据缺失的话这里会报错，比如只有一个位置数据---------------
                    trajectory_point['x_id'] = location[0]
                    trajectory_point['y_id'] = location[1]
                else:
                    is_exist = False
                    break
                trajectory_points.append(trajectory_point)
            if(is_exist):
                trajectory['points'] = trajectory_points
                trajectory['length'] = len(trajectory_points)

                trajectories.append(trajectory)
        f.close()
        #with open('somefile.txt', 'wt') as f:
            #f.write(str(trajectories))
    return trajectories

def Get_taxi_trajectories():
    trajectory_data_url = 'static/data/taxi_trajectories_smallsmallsmall.txt'
    trajectories = []
    with open(trajectory_data_url, 'rt') as f:
        for line in f:
            # data:460005731942647;05.91,26616;50:07.08,26616;13.27,26617;14.38,26617;
            trajectory_inf = line.split(':') # 按照冒号分割，前面是名字，后面是轨迹点
            if (len(trajectory_inf) !=2):
                continue
            trajectory = {}
            trajectory['id'] = trajectory_inf[0]
            trajectory_points = []  # 用于保存trajectory的各个点的位置和时间
            original_trajectory_points = trajectory_inf[1].split(';')# 需要判断轨迹点的数目，如果小于10就丢弃
            for i in range(len(original_trajectory_points)-1):  # 将每一个点保存,最后一个是\n，需要去掉
                trajectory_point = {}
                this_point = original_trajectory_points[i].split(',')
                #print(this_point)
                trajectory_point['x'] = float(this_point[0])
                trajectory_point['y'] = float(this_point[1])# 如果有数据缺失的话这里会报错，比如只有一个位置数据---------------
                trajectory_points.append(trajectory_point)
            trajectory['points'] = trajectory_points
            trajectory['length'] = len(trajectory_points)
            trajectories.append(trajectory)
        f.close()
        #with open('somefile.txt', 'wt') as f:
            #f.write(str(trajectories))
    return trajectories

def distance(point1,point2):
    try:
        lon1 = abs(point1['x'])
        lat1 = abs(point1['y'])
        lon2 = abs(point2['x'])
        lat2 = abs(point2['y'])
    except:
        print(point1)
        print(point2)
    hsinx=sin((lon1-lon2)*0.5);
    hsiny=sin((lat1-lat2)*0.5);
    h=hsiny*hsiny+(cos(lat1)*cos(lat2)*hsinx*hsinx);
    return 2*atan2(sqrt(h),sqrt(1-h))*6367000;


def matching_points(trajectories):
    #采用高级方式匹配轨迹点
    trajectories_matching_points_set = []#trajectories_matching_points[x][y]表示轨迹x和轨迹y的匹配轨迹点集合
    print(len(trajectories))
    for x in range(len(trajectories)):
        print(x)
        trajectory_matching_points_set = []#对第x个轨迹，计算与其他所有轨迹的匹配轨迹点
        for y in range(x+1,len(trajectories)):
            i = 0
            j = 0
            matching_stack = [(0,0)]
            trajectory_x=trajectories[x]
            trajectory_y=trajectories[y]
            while (i+1<trajectory_x['length']) and (j+1<trajectory_y['length']):
                d = [[float('inf'),float('inf'),float('inf')],[float('inf'),float('inf'),float('inf')],[float('inf'),float('inf'),float('inf')]]

                for k in range(3):#0,1,2
                    for n in range(3):
                        if (i+k < trajectory_x['length']) and (j+n < trajectory_y['length']):
                            d[k][n] = distance(trajectory_x['points'][i+k],trajectory_y['points'][j+n])
                if d[0][1] < d[0][0] and d[0][1] <= min(d[1][0],d[1][1],d[1][2],d[2][1]):
                    matching_stack.pop()
                    # j没人跟他对应，只能去跟i-1对应，但是要保证i-1没人对应
                    if i-1>0:
                        (a, b) = matching_stack[len(matching_stack) - 1]
                        if a != i - 1:  # 这一部分还需要商榷，到底要不要j去跟i-1对应-------------------------------------
                            matching_stack.append((i - 1, j))
                    matching_stack.append((i,j+1))
                else:
                    if d[1][0] < d[0][0] and d[1][0] <= min(d[0][1],d[1][1],d[1][2],d[2][1]):
                        matching_stack.pop()
                        #i没人跟他对应，只能去跟j-1对应，但是要保证j-1没人对应
                        if j-1>0:
                            (a, b) = matching_stack[len(matching_stack) - 1]
                            if b != j - 1:  # 这一部分还需要商榷，到底要不要i去跟j-1对应-------------------------------------
                                matching_stack.append((i, j - 1))
                        matching_stack.append((i+1,j))
                    else:
                        if d[1][2] < d[2][2] and d[1][2] <= min(d[0][1],d[1][0],d[1][1],d[2][1]):
                            matching_stack.append((i+1,j+2))
                        else:
                            if d[2][1] < d[2][2] and d[2][1] <= min(d[0][1], d[1][0], d[1][1], d[1][2]):
                                matching_stack.append((i + 1, j))
                            else:
                                matching_stack.append((i + 1, j+1))
                (i,j) = matching_stack[len(matching_stack)-1]
            trajectory_matching_points_set.append(matching_stack)
        if len(trajectory_matching_points_set)!=0:
            trajectories_matching_points_set.append(trajectory_matching_points_set)
    #with open('trajectories_matching_points_set.txt', 'wt') as f:
        #f.write(str(trajectories_matching_points_set))
    return trajectories_matching_points_set

class Similarity:
    def real_distance(self,parameters):#轨迹间对应轨迹点的矩阵，二维数组表示
        trajectories = parameters['trajectories']
        trajectories_matching_points_set = parameters['trajectories_matching_points_set']
        trajectories_similarity_matrix = []
        for x in range(len(trajectories)):
            trajectories_similarity_array = []#保存该行的欧几里得距离
            for y in range(0,x):
                trajectories_similarity_array.append(trajectories_similarity_matrix[y][x])
            trajectories_similarity_array.append(0.0)
            for y in range(x+1,len(trajectories)):
                trajectory_x = trajectories[x]
                trajectory_y = trajectories[y]
                matching_points=trajectories_matching_points_set[x][y-x-1]
                similarity=0#用于保存两个轨迹的实际距离
                for (i,j) in matching_points:
                    try:
                        similarity = similarity + distance(trajectory_x['points'][i], trajectory_y['points'][j])
                    except:
                        print(matching_points)
                        print(len(trajectory_x['points']))
                        print(len(trajectory_y['points']))
                        print(i,j)
                        return 'not matching'
                trajectories_similarity_array.append(similarity)
            trajectories_similarity_matrix.append(trajectories_similarity_array)
        return trajectories_similarity_matrix
    def frechet_distance(self,parameters):
        trajectories=parameters['trajectories']
        trajectories_matching_points_set=parameters['trajectories_matching_points_set']
        trajectories_similarity_matrix = []
        for x in range(len(trajectories)):
            trajectories_similarity_array = []  # 保存该行的距离
            for y in range(0,x):
                trajectories_similarity_array.append(trajectories_similarity_matrix[y][x])
            trajectories_similarity_array.append(0.0)
            for y in range(x+1,len(trajectories)):
                trajectory_x = trajectories[x]
                trajectory_y = trajectories[y]
                matching_points = trajectories_matching_points_set[x][y-x-1]
                max_distance = 0.0  # 用于保存两个轨迹的Frechet distance,找最大值
                for (i, j) in matching_points:
                    this_distance = distance(trajectory_x['points'][i], trajectory_y['points'][j])
                    if max_distance < this_distance:
                        max_distance = this_distance
                trajectories_similarity_array.append(max_distance)
            trajectories_similarity_matrix.append(trajectories_similarity_array)
        return trajectories_similarity_matrix
    def hausdorff_distance(self,parameters):
        trajectories = parameters['trajectories']
        trajectories_similarity_matrix = []
        for x in range(len(trajectories)):
            trajectories_similarity_array = []  # 保存该行的距离
            print(x)
            for y in range(0,x):
                trajectories_similarity_array.append(trajectories_similarity_matrix[y][x])
            trajectories_similarity_array.append(0.0)
            for y in range(x+1,len(trajectories)):
                trajectory_x = trajectories[x]['points']
                trajectory_y = trajectories[y]['points']
                max_distance = 0.0#在最短距离里面找一个最大值
                print('y',y)
                print(len(trajectory_x))
                print(len(trajectory_y))
                for a in range(len(trajectory_x)):
                    min_distance = -1#float('inf')#首先找到与另一个点集的最短距离
                    for b in range(len(trajectory_y)):
                        this_distance = distance(trajectory_x[a],trajectory_y[b])
                        if min_distance > this_distance:
                            min_distance = this_distance
                    if max_distance < min_distance or(min_distance==-1):
                        max_distance = min_distance
                trajectories_similarity_array.append(max_distance)
            trajectories_similarity_matrix.append(trajectories_similarity_array)
        return trajectories_similarity_matrix
    def lcss(self,parameters):
        trajectories = parameters['trajectories']
        trajectories_similarity_matrix = []
        for x in range(len(trajectories)):
            trajectories_similarity_array = []  # 保存该行的距离
            print(x)
            for y in range(0,x):
                trajectories_similarity_array.append(trajectories_similarity_matrix[y][x])
            trajectories_similarity_array.append(0.0)
            for y in range(x+1,len(trajectories)):
                print('y',y)
                trajectory_x = trajectories[x]['points']
                trajectory_y = trajectories[y]['points']
                trajectory_x_len = len(trajectory_x)
                trajectory_y_len = len(trajectory_y)
                c = [[0 for i in range(trajectory_y_len + 1)] for j in range(trajectory_x_len + 1)]
                flag = [[0 for i in range(trajectory_y_len + 1)] for j in range(trajectory_x_len + 1)]
                for i in range(trajectory_x_len):
                    for j in range(trajectory_y_len):
                        if trajectory_x[i] == trajectory_y[j]:
                            c[i + 1][j + 1] = c[i][j] + 1
                            flag[i + 1][j + 1] = 'ok'
                        elif c[i + 1][j] > c[i][j + 1]:
                            c[i + 1][j + 1] = c[i + 1][j]
                            flag[i + 1][j + 1] = 'left'
                        else:
                            c[i + 1][j + 1] = c[i][j + 1]
                            flag[i + 1][j + 1] = 'up'
                this_similarity = Lcss(flag, trajectory_x, trajectory_x_len, trajectory_y_len)
                trajectories_similarity_array.append(this_similarity)
            trajectories_similarity_matrix.append(trajectories_similarity_array)
        return trajectories_similarity_matrix

    def dtw(self,parameters):
        trajectories = parameters['trajectories']
        trajectories_similarity_matrix = []
        for x in range(len(trajectories)):
            trajectories_similarity_array = []  # 保存该行的距离
            print(x)
            for y in range(0,x):
                trajectories_similarity_array.append(trajectories_similarity_matrix[y][x])
            trajectories_similarity_array.append(0.0)
            for y in range(x+1,len(trajectories)):
                trajectory_X = trajectories[x]['points']
                trajectory_Y = trajectories[y]['points']
                trajectory_X_len = len(trajectory_X)
                trajectory_Y_len = len(trajectory_Y)
                M = [[distance(trajectory_X[i], trajectory_Y[j]) for i in range(trajectory_X_len)] for j in range(trajectory_Y_len)]
                D = [[0 for i in range(trajectory_X_len + 1)] for j in range(trajectory_Y_len + 1)]
                D[0][0] = 0
                for i in range(1, trajectory_X_len + 1):
                    D[0][i] = float('inf')
                for j in range(1, trajectory_Y_len + 1):
                    D[j][0] = float('inf')
                for j in range(1, trajectory_Y_len + 1):
                    for i in range(1, trajectory_X_len + 1):
                        D[j][i] = M[j - 1][i - 1] + min(D[j - 1][i], D[j][i - 1], D[j - 1][i - 1] + M[j - 1][i - 1])
                trajectories_similarity_array.append(D[trajectory_Y_len][trajectory_X_len])
            trajectories_similarity_matrix.append(trajectories_similarity_array)
        return trajectories_similarity_matrix

def Lcss(flag, a, i, j):
    if i == 0 or j == 0:
        return 0
    if flag[i][j] == 'ok':
        return 1+Lcss(flag, a, i - 1, j - 1)
    elif flag[i][j] == 'left':
        return Lcss(flag, a, i, j - 1)
    else:
        return Lcss(flag, a, i - 1, j)












class ClusterWay:
    def KMeans(self, parameters):
        result={}
        default_cluster = 4
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('cluster') is not None:
            default_cluster = int(parameters['cluster'])
        model = skc.KMeans(n_clusters=default_cluster)
        clustering = model.fit(data)
        result['labels'] = clustering.labels_
        return result

    def MiniBatchKMeans(self, parameters):  # data, n_clusters, batch_size):
        result = {}
        default_cluster = 3
        default_batch_size = 3
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('cluster') is not None:
            default_cluster = int(parameters['cluster'])
        if parameters.get('batch_size') is not None:
            default_batch_size = int(parameters['batch_size'])
        model = skc.MiniBatchKMeans(n_clusters=default_cluster, batch_size=default_batch_size)
        clustering = model.fit(data)
        result['labels'] = clustering.labels_
        return result

    def MeanShift(self, parameters):  # data, bandwidth):
        result = {}
        default_bandwidth = 3
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('bandwidth') is not None:
            default_bandwidth = int(parameters['bandwidth'])
        model = skc.MeanShift(bandwidth=default_bandwidth, bin_seeding=True)
        clustering = model.fit(data)
        result['labels'] = clustering.labels_
        return result

    def AffinityPropagation(self, parameters):  # data, damping, preference):
        result = {}
        default_damping = 0.7  # 0.5----1
        default_preference = 3
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('damping') is not None:
            default_damping = float(parameters['damping'])
        if parameters.get('preference') is not None:
            default_preference = int(parameters['preference'])
        model = skc.AffinityPropagation(damping=default_damping, preference=default_preference)
        clustering = model.fit(data)
        result['labels'] = clustering.labels_
        return result

    def Birch(self, parameters):  # data, threshold, branching_factor):
        result = {}
        default_threshold = 3
        default_branching_factor = 3
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('threshold') is not None:
            default_threshold = int(parameters['threshold'])
        if parameters.get('branching_factor') is not None:
            default_branching_factor = int(parameters['branching_factor'])
        model = skc.Birch(threshold=default_threshold, branching_factor=default_branching_factor)
        clustering = model.fit(data)
        result['labels'] = clustering.labels_
        return result

    def DBSCAN(self, parameters):  # data, eps, min_samples):
        result = {}
        default_eps = 3
        default_min_samples = 3
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('eps') is not None:
            default_eps = float(parameters['eps'])
        if parameters.get('min_samples') is not None:
            default_min_samples = int(parameters['min_samples'])
        model = skc.DBSCAN(eps=default_eps, min_samples=default_min_samples)
        clustering = model.fit(data)
        result['labels'] = clustering.labels_
        return result

    def HDBSCAN(self, parameters):  # data, min_cluster_size, min_samples, alpha, cluster_selection_method):
        result = {}
        default_min_cluster_size = 3
        default_min_samples = 3
        default_alpha = 0.5  # 大于1的float
        default_cluster_selection_method = "eom"  # "eom", "leaf"
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('min_cluster_size') is not None:
            default_min_cluster_size = int(parameters['min_cluster_size'])
        if parameters.get('min_samples') is not None:
            default_min_samples = int(parameters['min_samples'])
        if parameters.get('alpha') is not None:
            default_alpha = float(parameters['alpha'])
        if parameters.get('cluster_selection_method') is not None:
            default_cluster_selection_method = str(parameters['cluster_selection_method'])
        model = skc.HDBSCAN(min_cluster_size=default_min_cluster_size, min_samples=default_min_samples, alpha=default_alpha,
                        cluster_selection_method=default_cluster_selection_method, allow_single_cluster=True)
        clustering = model.fit(data)
        result['labels'] = clustering.labels_
        return result

    def GaussianMixture(self, parameters):  # data, n_clusters, cov_type):
        result = {}
        default_cluster = 3
        default_covariance_type = 'spherical'  # ['spherical', 'tied', 'diag', 'full']
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('cluster') is not None:
            default_cluster = int(parameters['cluster'])
        if parameters.get('covariance_type') is not None:
            default_covariance_type = str(parameters['covariance_type'])
        model = skc.GaussianMixture(n_components=default_cluster, covariance_type=default_covariance_type)
        clustering = model.fit(data)
        labels = model.predict(data)
        #result['clustering'] = clustering
        result['labels'] = labels
        return result

    def Hierarchical(self, parameters):  # data, method):
        result = {}
        default_method = 'single'
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        distance = skc.hierarchy.distance.pdist(data, 'euclidean')
        if parameters.get('cluster_method') is not None:
            default_method = str(parameters['cluster_method'])
        linkage = skc.hierarchy.linkage(distance, method=default_method)
        clustering = skc.hierarchy.fcluster(linkage, t=1, criterion='inconsistent')
        labels = pd.Series(clustering)
        #result['clustering'] = clustering
        result['labels'] = labels
        return result

    def AgglomerativeClustering(self, parameters):  # data, n_clusters):
        result = {}
        default_cluster = 3
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('cluster') is not None:
            default_cluster = int(parameters['cluster'])
        model = skc.AgglomerativeClustering(n_clusters=default_cluster)
        clustering = model.fit(data)
        result['clustering'] = clustering
        return result

    def SpectralClustering(self, parameters):  # data, n_clusters):
        result = {}
        default_cluster = 3
        data = np.array(parameters['data'])
        data = preprocessing.MinMaxScaler().fit_transform(data)
        if parameters.get('cluster') is not None:
            default_cluster = int(parameters['cluster'])
        model = skc.SpectralClustering(n_clusters=default_cluster)
        clustering = model.fit(data)
        # metrics.calinski_harabaz_score(data, clustering.labels_)
        result['clustering'] = clustering
        return result

'''
trajectories = Get_trajectories()
trajectories_matching_points_set = matching_points(trajectories)
Similarity=Similarity()
Similarity_matrix=Similarity.euclidean_Similarity(trajectories,trajectories_matching_points_set)
ClusterWay=ClusterWay()
parameters = {}
parameters['data'] = Similarity_matrix
cluster_result = ClusterWay.KMeans(parameters)
print(cluster_result['labels'])
'''







#def Euclidean():#欧几里得距离计算轨迹间的相似度









'''
trajectory_paths = open(trajectory_data_url)
trajectories = []
trajectory_line = trajectory_paths.readline()#轨迹读取的流对象,按行读取
print('222')
count = 0
while trajectory_line:
    if count==0:
        print(trajectory_line)
        break
    else:
        print('222')
        break
    trajectory_line_inf = trajectory_line.split('')
    trajectory = []
    trajectory['id'] = trajectory_line[0]
'''
