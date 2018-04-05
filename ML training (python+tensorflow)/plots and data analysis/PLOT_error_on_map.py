######### Suppresses warnings :D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#########


import pickle
import numpy as np
import matplotlib.pyplot as plt


data_c = True



print("Loading the predictions...", end = '')
with open('predictions.pkl', 'rb') as f:
    distance_output, all_labels, non_zeroes = pickle.load(f)
 
len_predictions = len(distance_output)

print("({0} predictions loaded)".format(len_predictions))
print("x range: {0} - {1}".format(all_labels[:,0].min(), all_labels[:,0].max()))
print("y range: {0} - {1}".format(all_labels[:,1].min(), all_labels[:,1].max()))
 
avg_distance = np.mean(distance_output)
sorted_distance = np.sort(distance_output)
distance_95 = sorted_distance[int(0.95 * len_predictions)]
print('\n\nTest distance (m)= {0:.4f},   95% percentile (m) = {1:.4f}'.format(avg_distance, distance_95))



#creates empty data structures   ->ROW (X) MAJOR i.e. row1 -> row2 -> row3
position_entries = [[] for i in range(401*401)]
position_error = [100] * (401*401)
position_error = np.asarray(position_error)

#for each prediction: gets the position -> gets the matrix index -> stores the error
print("\nSorting the predictions by position...") 
for i in range(len_predictions):
    x = all_labels[i,0] * 400   #[0,1] -> [0,400]
    y = (1.0-all_labels[i,1]) * 400       #flips y
    
    matrix_index = int(x * 401 + y)
    
    position_entries[matrix_index].append(distance_output[i])
    
#gets each position's average   (if it has no entries, sets as default value)
print("\nAveraging the results by position...")
for i in range(401*401):
    if len(position_entries[i]) > 0:
        position_error[i] = np.mean(position_entries[i])


        
position_error_2D = position_error.reshape(401, 401)


############################################
#DATA CORRECTION!
# During the ray-tracing, I had to manually define the "ground" on multiple points
# It seems that I missed a few points </3
# Result: some rays passed UNDER the building, leading to existing entries under the buildings :(
# However, this should not influence the real results: the rays were limited to 6 reflections, and,
# to go under a building, the ray must have had a steep angle, bounding under the building (so it couldn't have gone much further away)

# If you are reusing my data: sorry :(

if data_c:

    print("\nFixing the simulation bugs...")
    
    #defines the ray casting algorithm (https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon)
    def ray_casting(n_vertex, vertices_x, vertices_y, point_x, point_y):
        """vertices_x and _y are lists with n_vertex entries, depicting the polygon"""
    
        #c = crosses; even crosses = outside the polygon, odd = inside
        c = 0
        j = n_vertex - 1
        i = 0
        
        while(i < n_vertex):
            
            if((vertices_y[i] > point_y) != (vertices_y[j] > point_y)):
                if(point_x < (vertices_x[j] - vertices_x[i]) * (point_y - vertices_y[i]) / (vertices_y[j] - vertices_y[i]) + vertices_x[i] ):
                    c += 1
            
            #loop update conditions
            j = i
            i += 1
            
            
        
        if(c % 2 == 0):
            inside = False
        else:
            inside = True
        
    
        return(inside)
        
        
    poligons = [[4, [175,140,171,206], [153,209,229,175]]]
    poligons.append([6, [268,301,331,320,306,277], [290,238,255,262,253,292]])
    poligons.append([4, [184,208,225,206], [138,102,113,150]])
    poligons.append([4, [172,213,223,183], [224,249,233,209]])
    poligons.append([6, [225, 230, 231, 250, 268, 239], [260, 256, 250, 219, 231, 272]])
    
    n_poligions = len(poligons)
    
    
    #tests all points
    for i in range(401*401):
    
        j = 0
        inside = False
        
        point_x = int(i % 401)
        point_y = int((i - point_x)/401)
        
        while((j < n_poligions) and (inside == False)):
        
            poligon = poligons[j]
            
            n_vertex = poligon[0]
            vertices_x = poligon[1]
            vertices_y = poligon[2]
            
            inside = ray_casting(n_vertex, vertices_x, vertices_y, point_x, point_y)
        
            j += 1 
        
        
        if inside:
            position_error_2D[point_x, point_y] = 100.0
    

############################################



#plots the thing
cax = plt.imshow(np.transpose(position_error_2D), vmin = 0.0, vmax = 50)
plt.colorbar(cax)


# plt.savefig('error_vs_nonzero.pdf', format='pdf')
plt.show()