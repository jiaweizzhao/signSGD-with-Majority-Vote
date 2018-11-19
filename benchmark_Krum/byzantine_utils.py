import torch
import numpy as np

def majority_vote(gradient_list):
	for i,gradient in enumerate(gradient_list):
		if i==0:
			sum_signs = [torch.sign(g) for g in gradient_list[0]]
		else:
			sum_signs = [torch.sign(g) + s for (g,s) in zip(gradient_list[i],sum_signs)]
	vote = [torch.sign(s) for s in sum_signs]
	return vote

def distance(grad1, grad2):

    #torch version
    dist = torch.dist(grad1, grad2, 2)

    return dist

    '''
	dist = 0
	for g1,g2 in zip(grad1,grad2):
		dist += torch.norm(g1-g2)**2
    '''

def sum_grads(gradient_list):

    
	for i,gradient in enumerate(gradient_list):
		if i==0:
			sum_grad = gradient
		else:
			sum_grad += gradient
    
	return sum_grad

def krum(gradient_list, f, multi = True):
    score_list = []
    for i,vi in enumerate(gradient_list):
        dist_list = []
        for j,vj in enumerate(gradient_list):
            dist_list.append(distance(vi,vj))
        dist_list.sort()
        truncated_dist_list = dist_list[:-(f+1)]
        score = sum(truncated_dist_list)
        score_list.append(score)
    sorted_score_indices = np.argsort(np.array(score_list))
    if multi:
    	return sum_grads([gradient_list[score_idx] for score_idx in sorted_score_indices[:-f]])	
    else:
    	return gradient_list[sorted_score_indices[0]]

def update_params(params,gradient,lr):
	for p,g in zip(params,gradient):
		p.data.add_(-lr, g)