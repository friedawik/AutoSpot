import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
#from skimage import data, feature, color
from IPython import embed


class SpotDetector:
    def __init__(self, image, contour_levels, min_area, max_area, min_circularity, z_min_dist):
        self.image = image
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.z_min_dist = z_min_dist
        self.contour_levels = contour_levels
        self.contours = plt.contour(self.image, levels=contour_levels)    
        self.closed_paths = {}

    def make_contour_object(self):
        plt.rcParams['figure.figsize'] = [10, 5] 
        self.contours = plt.contour(self.image, levels=self.contour_levels)
        plt.close()
        
    def get_area(self, vertices):
        x = vertices[:, 0]
        y = vertices[:, 1]
        # shoelace algorithm
        polygon_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return polygon_area

    def get_perimeter(self, vertices):
        return np.sum(np.sqrt(np.sum(np.diff(vertices, axis=0)**2, axis=1)))

    def get_circularity(self, vertices, area):
        perimeter = self.get_perimeter(vertices)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return circularity
    
    def get_bbox(self, path):
        # Get the extents of the path
        bounds = path.get_extents()
        # Add padding
        pad = 0
        # Extract the coordinates in xyxy format
        x_min, y_min = bounds.min
        x_max, y_max = bounds.max
        bbox = (x_min-pad, 
                y_min-pad, 
                x_max+pad, 
                y_max+pad)
        return bbox

    def get_closed_contours(self):
        self.make_contour_object()
        count = 0
 
        for level, path in zip(self.contours.levels, self.contours.get_paths()):
        #for level, collection in enumerate(self.contours.collections):
       
            #for path in collection.get_paths():
        
            all_codes_arr = path.codes
            all_verts_arr = path.vertices
           
            
            #start_ind = np.where(np.atleast_1d(all_codes_arr)==1)[0]
            if all_codes_arr is None:
                continue
            start_ind = np.where(all_codes_arr==1)[0] # Each new path starts with vertex==1
            for i in range(len(start_ind)): 
                # Get the last contour in list
                if i == len(start_ind)-1:
                    if all_codes_arr[-1]==79: # Each closed path ends with vertex==79
                        single_codes = all_codes_arr[start_ind[i]::]
                        single_verts = all_verts_arr[start_ind[i]::]
                        area = self.get_area(single_verts)
                        if self.min_area < area < self.max_area:
                            circularity = self.get_circularity(single_verts, area)
                            if circularity > self.min_circularity:
                                single_path = Path(single_verts, single_codes )
                                centroid = self.get_centroid(single_path)
                                bbox = self.get_bbox(single_path)                        
                                self.closed_paths[count]={'level':level, 'path':single_path, 'area':area, 'centroid':centroid, 'bbox':bbox}
                                count += 1
                # Get all other contours
                elif all_codes_arr[start_ind[i+1]-1] == 79:
                    single_codes = all_codes_arr[start_ind[i]:start_ind[i+1]]
                    single_verts = all_verts_arr[start_ind[i]:start_ind[i+1]]
                    area = self.get_area(single_verts)
                    if self.min_area < area < self.max_area:
                        circularity = self.get_circularity(single_verts, area)
                        if circularity > self.min_circularity:
                            single_path = Path(single_verts, single_codes )
                            centroid = self.get_centroid(single_path)
                            bbox = self.get_bbox(single_path)                   
                            self.closed_paths[count]={'level':level, 'path':single_path, 'area':area, 'centroid':centroid, 'bbox':bbox}
                            count += 1

    

    def get_base(self):
        """ Function that loops through a sorted dictionary (descending area) to check if centroid is inside
        a base contour. If it is, the corresponiding path is saved as a base path and the entry is removed from the list. The
         loop continues while there are still  """
        self.get_closed_contours()   
        paths = self.closed_paths.copy()
        paths_temp = {k: v for k, v in sorted(paths.items(), key=lambda item: item[1]['area'], reverse=True)}
        new_dict = {}
        count_base = 0
        print(f'There are {len(paths_temp)} contours')
        while len(paths_temp)>0:

            first_key = next(iter(paths_temp.keys()))
            base_path = paths_temp[first_key]['path']
            bool_arr = np.ones(len(paths_temp), dtype=bool)
            bool_arr[0]=False
            count = 0
            double_count = 0
            single_contours = True
            for key, value in paths_temp.items():   
                centroid = value['centroid']
                if base_path.contains_point(centroid):
                    bool_arr[count]=False
                    base_level = paths_temp[first_key]['level']
                    check_level = value['level']
                    if base_level < check_level:
                       # if check_level-base_level > self.z_min_dist:
                        if single_contours: 
                            new_dict[first_key] = paths_temp[first_key] 
                            double_count += 1
                            single_contours = False
                count += 1
            filtered_data = {key: value for (key, value), valid in zip(paths_temp.items(), bool_arr) if valid}
            paths_temp = filtered_data
            count_base += 1
        return new_dict

    # # FIXME: what happens to base countors that include several features?
    # def get_features(self): 
    #     self.get_closed_contours()   
    #     paths = self.closed_paths.copy()
    #     paths_temp = {k: v for k, v in sorted(paths.items(), key=lambda item: item[1]['area'], reverse=True)}
    #     new_dict = {}
    #     count_base = 0
    #     print(f'There are {len(paths_temp)} contours')
    #     while len(paths_temp)>0:
    #         first_key = next(iter(paths_temp.keys()))
    #         base_path = paths_temp[first_key]['path']
    #         bool_arr = np.ones(len(paths_temp), dtype=bool)
    #         bool_arr[0]=False
    #         count = 0
    #         single_contours = True
    #         temp_count = 0
    #         dict_temp = {}
    #         for key, value in paths_temp.items(): 
    #             centroid = value['centroid']
    #             if base_path.contains_point(centroid):         
    #                 bool_arr[count]=False
    #                 base_level = paths_temp[first_key]['level']
    #                 check_level = value['level']
    #                 if base_level <= check_level:
    #                     dict_temp[temp_count] = paths_temp[key]          
    #                 temp_count += 1        
    #             count += 1
    #         if len(dict_temp)>1:
    #             new_dict[count_base] = dict_temp
    #             count_base += 1
               
                
    #         filtered_data = {key: value for (key, value), valid in zip(paths_temp.items(), bool_arr) if valid}
    #         paths_temp = filtered_data
            
    #     return new_dict    
    


    # FIXME: what happens to base countors that include several features?
    def get_features(self): 
        self.get_closed_contours()   
        paths = self.closed_paths.copy()
        paths_temp = {k: v for k, v in sorted(paths.items(), key=lambda item: item[1]['area'], reverse=True)}
        new_dict = {}
        count_base = 0
        print(f'There are {len(paths_temp)} contours')
        while len(paths_temp)>0:
            first_key = next(iter(paths_temp.keys()))
            base_path = paths_temp[first_key]['path']
            bool_arr = np.ones(len(paths_temp), dtype=bool)
            bool_arr[0]=False
            count = 0
            temp_count = 0
            dict_temp = {}
            for key, value in paths_temp.items(): 
                centroid = value['centroid']      
                if base_path.contains_point(centroid):   
                    bool_arr[count]=False
                    base_level = paths_temp[first_key]['level']
                    check_level = value['level']
                    if base_level <= check_level:
                        dict_temp[temp_count] = paths_temp[key]        
                        temp_count += 1        
                count += 1
            

            if self.check_merged_spots(dict_temp):

                #FIXME: disabled this one to test filtering later
                bool_arr = np.ones(len(paths_temp), dtype=bool)
                bool_arr[0]=False

            else: 
                if len(dict_temp)>1:
                    new_dict[count_base] = dict_temp
                    count_base += 1
            
            filtered_data = {key: value for (key, value), valid in zip(paths_temp.items(), bool_arr) if valid}
            paths_temp = filtered_data
            
        return new_dict  

    def check_merged_spots(self, dict):
        levels = []
        for level in dict:
            levels.append(dict[level]['level'])
        has_duplicates = any(levels.count(value) > 1 for value in levels)
        return has_duplicates


    def get_centroid(self, path):
        centroid = path.vertices.mean(axis=0)
        return centroid
        
    def plot_spots(self):     
        spots = self.get_base()
        plt.rcParams['figure.figsize'] = [10, 5] 
        fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        axes[0].imshow(self.image, cmap = 'gray')
        axes[1].imshow(self.image, cmap = 'gray')

        for key in spots.keys():
            path = spots[key]['path']
            patch = patches.PathPatch(path, edgecolor='orange', fill=False, lw=1)
            axes[1].add_patch(patch)
        plt.show()

    def plot_features(self, features):     
        #features = self.get_features()
        
        plt.rcParams['figure.figsize'] = [10, 5] 
        fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        axes[0].imshow(self.image, cmap = 'gray')
        axes[1].imshow(self.image, cmap = 'gray')

        for key in features.keys():
            last_ind = len(features[key])-1
            diff_levels = features[key][last_ind]['level'] - features[key][0]['level'] 
   
            if  diff_levels > self.z_min_dist:
                path = features[key][0]['path']
                patch = patches.PathPatch(path, edgecolor='orange', fill=False, lw=1)
                axes[1].add_patch(patch)
        plt.savefig('test.png')
     
    def check_z_distance(self, features):
        new_dict = {}
        count = 0
        for key in features.keys():
            last_ind = len(features[key])-1
            diff_levels = features[key][last_ind]['level'] - features[key][0]['level'] 
           
   
            if  diff_levels > self.z_min_dist:
                new_dict[count] = features[key]
                count += 1
        return new_dict


    def plot_3d_features(self):     
        features = self.get_features()
        x = np.linspace(0, self.image.shape[1] - 1, self.image.shape[1])
        y = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
        x, y = np.meshgrid(x, y)
        points = np.vstack((x.ravel(), y.ravel())).T

        # Create a figure and a 3D axis
        plt.rcParams['figure.figsize'] = [10, 10] 
        fig, ax = plt.subplots()

        masks = np.zeros((len(x),len(y)))
        for key in features.keys():
            last_ind = len(features[key])-1
            diff_levels = features[key][last_ind]['level'] - features[key][0]['level'] 
            if  diff_levels > self.z_min_dist:
                path = features[key][0]['path']
                inside = path.contains_points(points)
                inside = inside.reshape((len(x), len(y)))
          
                masks[inside] = 1

        plt.imshow(masks)
        

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        #surf = ax.plot_surface(x, y, self.image,alpha=0.6)
        #non_masks = np.where(masks==0)
        #non_masked_data = np.where(non_masks, self.image, np.nan) 
        surf = ax.plot_surface(x, y, self.image,alpha=0.5)
        #non_mask_surf = ax.plot_surface(x, y, non_masked_data, color='blue')
        
        masked_data = np.where(masks, self.image, np.nan)  # Set unmasked areas to NaN
        mask_surf = ax.plot_surface(x, y, masked_data, color='red')
        # Set labels
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Pixel Value (Z Axis)')
        plt.show()

    def plot_centroids(self):
        plt.rcParams['figure.figsize'] = [10, 5] 
        fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        axes[0].imshow(self.image, cmap = 'gray')
        axes[1].imshow(self.image, cmap = 'gray')
        
        for l, level in enumerate(self.centroid_list):
            x = [i[0] for i in level]
            y = [i[1] for i in level]
            axes[1].scatter(x, y, color= 'b', s = 2)

        plt.show()