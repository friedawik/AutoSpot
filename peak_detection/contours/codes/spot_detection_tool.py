import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
#from skimage import data, feature, color
from IPython import embed
from collections import defaultdict


class SpotDetector:
    def __init__(self, image, contour_levels, min_area, max_area, min_circularity, z_min_dist):

        """ 
        Code to detect stacked contours from the output of the plt.contour() function. 
        Any overlapping, closed contours are saved as a dictionary and the centroid of
        the top contour is returned as the peak.

        contour_levels:     The number of contour levels to generate. Higher values create contours at more frequent pixel values.
        min_area:           The minimum area a contour must have to be considered valid.
                            Contours smaller than this will be ignored.
        max_area:           The maximum area a contour can have to be considered.
                            Contours larger than this will be ignored.
        min_circularity:    The minimum circularity a contour must have to be considered valid.
                            Circularity is calculated by shoelace algorithm.
                            A perfect circle has a circularity of 1.0.
        z_min_dist:         The minimum vertical distance between the base and peak level to be considered.
    
        """

        # Tunable parameters
        self.image = image
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.z_min_dist = z_min_dist
        self.contour_levels = contour_levels
        
        # Dictionaries for storing paths and peaks    
        self.closed_paths = {}
        self.peak_dict = {}

        # Make contourset and fill dictionaries for each class instance
        self.make_contour_object() # Makes the plt quadContourSet
        self.get_closed_contours()  # Saves all closed contours in self.closed_paths
        self.get_features()         # Saves all peaks in self.peak_dict 

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
      
    def get_centroid(self, path):
        centroid = path.vertices.mean(axis=0)
        return centroid
    
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
        """
        The Path object contains vertices and codes. In the following function the single closed contours are found by
        looking for code MOVETO = np.uint8(1) followed by CLOSEPOLY = np.uint8(79). All other paths are discarded.
        """
        count = 0
        for level, path in zip(self.contours.levels, self.contours.get_paths()):
            all_codes_arr = path.codes
            all_verts_arr = path.vertices
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
        """ 
        Function that loops through a sorted dictionary (descending area) to check if centroid is inside
        a base contour. If it is, the corresponiding path is saved as a base path and the entry is removed from the list. The
         loop continues while there are still entries left and the largest of all contours will be saved as base path. 
         """
          
        paths = self.closed_paths.copy()
        paths_temp = {k: v for k, v in sorted(paths.items(), key=lambda item: item[1]['area'], reverse=True)}
        new_dict = {}
        count_base = 0
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

  
    


    # FIXME: what happens to base countors that include several features?
    def get_features(self): 
        """
        This function loops through the self.closed_paths and checks if the centroid of any other paths is within its area.
        If it is, the centroid of the top contour will be choosen as peak and returned in a dictionary with all peaks. 
        Subpeaks are also found by looking for 

        """
        paths = self.closed_paths.copy()
        paths_temp = {k: v for k, v in sorted(paths.items(), key=lambda item: item[1]['area'], reverse=True)}
        new_dict = {}
        count_base = 0
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
                    base_level = paths_temp[first_key]['level'] # pixel value of base level
                    check_level = value['level']                # pixel value of point
                    if base_level <= check_level:               # only keep if point above base (keep peaks, exclude bottoms)
                        dict_temp[temp_count] = paths_temp[key]        
                        temp_count += 1        
                count += 1
            
            
      
            peak_dict = self.get_peak(dict_temp, base_level)
            if peak_dict:
            #for i, row in  enumerate(peak_dict):
             #   embed()
              #  new_dict[count_base] = peak_dict[i] #FIXME we are losing some values here, wrong way of adding together
               # count_base += 1

                new_dict[count_base] = peak_dict[0] #FIXME we are losing some values here, wrong way of adding together
                count_base += 1
            # if self.check_merged_spots(dict_temp):

            #     #FIXME: disabled this one to test filtering later
            #     bool_arr = np.ones(len(paths_temp), dtype=bool) # this will make sure that all contours are added not only base contour
            #     bool_arr[0]=False

            # else: 
            #     if len(dict_temp)>1:
            #         new_dict[count_base] = dict_temp
            #         count_base += 1
            
            filtered_data = {key: value for (key, value), valid in zip(paths_temp.items(), bool_arr) if valid}
            paths_temp = filtered_data
            
        self.peak_dict = new_dict 
        return self.peak_dict 

    
    def get_peak(self, contour_dict, base_level):
        if not contour_dict:
            return None
        peak_inds = []
        peak_dict = {}

        max_ind = max(contour_dict, key=lambda k: contour_dict[k]['level'])
        peak_inds.append(max_ind)
        sub_inds = self.get_sub_peaks(contour_dict)

        for key, value in sub_inds.items():
            peak_inds.extend(value)
        
        for i, ind in enumerate(peak_inds):
            z_dist = contour_dict[ind]['level'] - base_level
            if z_dist >= self.z_min_dist:
                peak_dict[i] = contour_dict[ind]
                peak_dict[i]["z_dist"] = z_dist
                peak_dict[i]["base_level"] = base_level
       
        return peak_dict

        
        
    def get_sub_peaks(self, contour_dict):
        # FIXME: shoudl keep only top overlapping peak 
        level_groups = defaultdict(list)
        for key, value in contour_dict.items():
            level_groups[value['level']].append(key)
        return {level: keys for level, keys in level_groups.items() if len(keys) > 1}



    def plot_3d_features(self):     
        x = np.linspace(0, self.image.shape[1] - 1, self.image.shape[1])
        y = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
        x, y = np.meshgrid(x, y)
        points = np.vstack((x.ravel(), y.ravel())).T

        # Create a figure and a 3D axis
        plt.rcParams['figure.figsize'] = [10, 10] 
        fig, ax = plt.subplots()

        masks = np.zeros((len(x),len(y)))
        for key in self.peak_dict.keys():
            last_ind = len(self.peak_dict[key])-1
            diff_levels = self.peak_dict[key][last_ind]['level'] - self.peak_dict[key][0]['level'] 
            if  diff_levels > self.z_min_dist:
                path = self.peak_dict[key][0]['path']
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
        # embed()
        for key, value in self.peak_dict.items():
        # for l, level in enumerate(self.peak_dict):
         
            x =  value['centroid'][0]
            y =  value['centroid'][1]
            axes[1].scatter(x, y, color= 'r', s = 2)
        plt.show()
        # plt.savefig('../plots/centroids.png')
        

        # plt.savefig("../plots/centroids.png")