import math
import copy
import numpy as np

def ConvertToProb(input_list):
	'''Function to convert our i class list into a probability distribution'''
	new_list = []
	for i in range(len(input_list)):
		new_list.append(input_list[i]/float(sum(input_list)))
		
	return new_list

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
	
def Dataset_to_Distribution(dataset):
	mean, std = np.mean(dataset), np.std(dataset)
	x_axis_range = np.linspace(-3*std, 3*std, num=18)

	histogram_of_dataset1 = []
	for i in range(18):
		histogram_of_dataset1.append(gaussian(x_axis_range[i],mean,std))
		
	prob_distribution = ConvertToProb(histogram_of_dataset1)
	return prob_distribution

def battacharya(P,Q):
  sum_list=[]
  for i in range(len(P)):
    sum_list.append(math.sqrt(P[i]*Q[i]))
  
  return -math.log(sum(sum_list))
  
def KLDivergence(P,Q):
	'''Function to calculate KL divergence score	'''
	sum_list=[]
	for i in range(len(P)):
		try:
			sum_list.append(P[i]*math.log(P[i]/float(Q[i])))
		except:
			sum_list.append(0)
		
	return sum(sum_list)
	
def Relative_Distance(model,original_labels,test_data,shape,verbose=1):
	
	predictions_new_data = model.predict_classes(test_data)
	[img_rows,img_cols,colors] = shape
	
	correct_predictions=[]
	wrong_predictions=[]
	for i in range(len(predictions_new_data)):
	  if predictions_new_data[i] == original_labels[i]:
	    correct_predictions.append(i)
	  else:
	    wrong_predictions.append(i)
	if verbose ==1:
		print("Number of correctly predicted points: ",len(correct_predictions))

	A_points_indices=[]
	B_points_indices=[]
	C_points_indices=[]
	D_points_indices=[]
	E_points_indices=[]
	F_points_indices=[]
	G_points_indices=[]
	H_points_indices=[]
	I_points_indices=[]
	test=[]

	for i in correct_predictions:
		# TO DO: Change the logic here. Some undeclared variables are there.te
		test_point = test_data_1[i]
		test_point = np.reshape(test_point,(1,img_rows,img_cols,colors))
		confidence = model.predict(test_point)
		confidence = max(confidence[0]) 
		if confidence >= 0.99:
			A_points_indices.append(i)
		elif confidence >= 0.90 and confidence <= 0.98:
			B_points_indices.append(i)
		elif confidence >= 0.80 and confidence < 0.90:
			C_points_indices.append(i)
		elif confidence >= 0.75 and confidence < 0.8:
			D_points_indices.append(i)
		elif confidence >= 0.70 and confidence < 0.75:
			E_points_indices.append(i)
		elif confidence >= 0.65 and confidence < 0.7:
			F_points_indices.append(i)
		elif confidence >= 0.6 and confidence < 0.65:
			G_points_indices.append(i)
		elif confidence >= 0.55 and confidence < 0.6:
			H_points_indices.append(i)
		elif confidence >= 0.5 and confidence < 0.55:
			I_points_indices.append(i)
		else:
			test.append(i)
	if verbose==1:
		print("Number of perfect points found: ",len(A_points_indices))
		print("Number of B class points found: ",len(B_points_indices))
		print("Number of C class points found: ",len(C_points_indices))
		print("Number of D class points found: ",len(D_points_indices))
		print("Number of E class points found: ",len(E_points_indices))
		print("Number of F class points found: ",len(F_points_indices))
		print("Number of G class points found: ",len(G_points_indices))
		print("Number of H class points found: ",len(H_points_indices))
		print("Number of I class points found: ",len(I_points_indices))
		print(len(test))

	J_points_indices=[]
	K_points_indices=[]
	L_points_indices=[]
	M_points_indices=[]
	N_points_indices=[]
	O_points_indices=[]
	P_points_indices=[]
	Q_points_indices=[]
	R_points_indices=[]
	test=[]

	for i in wrong_predictions:
	  test_point = test_data_1[i]
	  test_point = np.reshape(test_point,(1,img_rows,img_cols,colors))
	  confidence = model.predict(test_point)
	  confidence = max(confidence[0]) 
	  if confidence >= 0.99:
	    J_points_indices.append(i)
	  elif confidence >= 0.90 and confidence <= 0.98:
	    K_points_indices.append(i)
	  elif confidence >= 0.80 and confidence < 0.90:
	    L_points_indices.append(i)
	  elif confidence >= 0.75 and confidence < 0.8:
	    M_points_indices.append(i)
	  elif confidence >= 0.70 and confidence < 0.75:
	    N_points_indices.append(i)
	  elif confidence >= 0.65 and confidence < 0.7:
	    O_points_indices.append(i)
	  elif confidence >= 0.6 and confidence < 0.65:
	    P_points_indices.append(i)
	  elif confidence >= 0.55 and confidence < 0.6:
	    Q_points_indices.append(i)
	  elif confidence >= 0.5 and confidence < 0.55:
	    R_points_indices.append(i)
	  else:
	    test.append(i)
	if verbose == 1:
		print("Number of J class points found: ",len(J_points_indices))
		print("Number of K class points found: ",len(K_points_indices))
		print("Number of L class points found: ",len(L_points_indices))
		print("Number of M class points found: ",len(M_points_indices))
		print("Number of N class points found: ",len(N_points_indices))
		print("Number of O class points found: ",len(O_points_indices))
		print("Number of P class points found: ",len(P_points_indices))
		print("Number of Q class points found: ",len(Q_points_indices))
		print("Number of IR class points found: ",len(R_points_indices))
		print(len(test))

	y_axis_1 = 		[len(A_points_indices),len(B_points_indices),len(C_points_indices),len(D_points_indices),len(E_points_indices),len(F_points_indices),len(G_points_indices),len(H_points_indices),len(I_points_indices),len(J_points_indices),len(K_points_indices),len(L_points_indices),len(M_points_indices),len(N_points_indices),len(O_points_indices),len(P_points_indices),len(Q_points_indices),len(R_points_indices)]
	return y_axis_1
