from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from django.http import HttpResponse
from trajectory_cluster.app.trajectory_cluster import *
import json
def index(request):
    global trajectories
    trajectories = Get_taxi_trajectories()
    print('finish')
    global trajectories_matching_points_set
    trajectories_matching_points_set = matching_points(trajectories)
    return render(request,'trajectory_cluster.html')
    #return HttpResponse('hello world')

@csrf_exempt
def cluster_way(request):
    #cluster_way=request.
    if request.method == 'POST':
        Similarity_parameters = {}
        Cluster_parameters = {}

        cluster_method = str(request.POST.get('cluster_method'))
        similarity_method = str(request.POST.get('similarity_method'))
        Similarity_parameters['trajectories'] = trajectories
        Similarity_parameters['trajectories_matching_points_set'] = trajectories_matching_points_set
        Similarity_matrix = getattr(Similarity(), similarity_method)(Similarity_parameters)
        Cluster_parameters['data'] = Similarity_matrix
        cluster_result = getattr(ClusterWay(), cluster_method)(Cluster_parameters)
        for i in range(len(trajectories)):
            trajectories[i]['labels'] = str(cluster_result['labels'][i])
        result = {'trajectories': trajectories, 'labels': cluster_result['labels']}
        return HttpResponse(json.dumps(result, ensure_ascii=False), content_type='application/json; charset=utf-8')
    else:
        return HttpResponse('error!')
