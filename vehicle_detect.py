import numpy as np
import pickle, cv2, glob, time
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

def read_image(img_path):
    return np.asarray(Image.open(img_path))

def hog_feature_extract(img,orient=9,pix_per_cell=8,cell_per_block=2,visualise=False,feature_vector=True):
    if visualise:
        hog_feature,hog_image = hog(img,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),cells_per_block=(cell_per_block,cell_per_block),transform_sqrt=False,visualise=visualise,feature_vector=feature_vector)
        return hog_feature,hog_image
    else:
        hog_feature = hog(img,orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),cells_per_block=(cell_per_block,cell_per_block),transform_sqrt=False,visualise=visualise,feature_vector=feature_vector)
        return hog_feature

def bin_spatial_feature_extract(img,size=(32,32)):
    ch1= cv2.resize(img[:,:,0],size).ravel()
    ch2 = cv2.resize(img[:,:,1],size).ravel()
    ch3 = cv2.resize(img[:,:,2],size).ravel()
    return np.hstack((ch1,ch2,ch3))

def color_hist_feature_extract(img,n_bins=30):
    ch1_hist = np.histogram(img[:,:,0],bins=n_bins)
    ch2_hist = np.histogram(img[:,:,1],bins=n_bins)
    ch3_hist = np.histogram(img[:,:,2],bins=n_bins)
    return np.concatenate((ch1_hist[0],ch2_hist[0],ch3_hist[0]))

def extract_all_features(img_paths,color_space='YCrCb',spatial_size=(32,32),n_bins=30,orient=9,pix_per_cell=8,cell_per_block=2,hog_channel='ALL',spatial_bool=True,hist_bool=True,hog_bool=True):
    features = []
    for img_path in img_paths:
        file_features = []
        image = read_image(img_path)
        if color_space == 'RGB':
            feature_image = np.copy(image)
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2LUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)
        
        if spatial_bool:
            spatial_features = bin_spatial_feature_extract(feature_image,size=spatial_size)
            file_features.append(spatial_features)
        if hist_bool:
            hist_features = color_hist_feature_extract(feature_image,n_bins=n_bins)
            file_features.append(hist_features)
        if hog_bool:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(hog_feature_extract(feature_image[:,:,channel],orient,pix_per_cell,cell_per_block,visualise=False,feature_vector=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = hog_feature_extract(feature_image[:,:,hog_channel],orient,pix_per_cell,cell_per_block,visualise=False,feature_vector=True)
            
            file_features.append(hog_features)
        
        features.append(np.concatenate(file_features))
        
    return features

def find_vehicles(img,y_start,y_stop,scale,svc,scaler,orient=9,pix_per_cell=8,cell_per_block=2,spatial_size=(32,32),n_bins=30,hog_channel='ALL',color_space='YCrCb',spatial_bool=True,hist_bool=True,hog_bool=True):
    img_convert = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb) # convert image to YCrCb
#     img_convert_normalized = img_convert.astype(np.float32)/255 # convert to float32 (since the original image is a uint8) and normalize image features
    img_cropped = img_convert[y_start:y_stop,:,:] # crop image
    
    # scale cropped image if scale isn't equal to 1
    if scale != 1:
        height,width = img_cropped.shape[:2]
        img_cropped = cv2.resize(img_cropped,(np.int(width/scale),np.int(height/scale)))
        
    ch1 = img_cropped[:,:,0]
    ch2 = img_cropped[:,:,1]
    ch3 = img_cropped[:,:,2]
    
    height,width = ch1.shape[:2]
    
    nx_blocks = np.int(width/pix_per_cell)+1
    ny_blocks = np.int(height/pix_per_cell)-1
    n_features_per_block = orient*cell_per_block**2
    
    window = 64
    n_blocks_per_window = np.int(window/pix_per_cell)-1
    cells_per_step = 2
    nx_steps = np.int((nx_blocks-n_blocks_per_window)/cells_per_step)
    ny_steps = np.int((ny_blocks-n_blocks_per_window)/cells_per_step)
    
    ch1_hog = hog_feature_extract(ch1,feature_vector=False)
    ch2_hog = hog_feature_extract(ch2,feature_vector=False)
    ch3_hog = hog_feature_extract(ch3,feature_vector=False)
    
    on_windows = []
    for x in range(nx_steps):
        for y in range(ny_steps):
            y_loc = y*cells_per_step
            x_loc = x*cells_per_step
                
            x_left = x_loc*pix_per_cell
            y_top = y_loc*pix_per_cell
            
            img_patch = cv2.resize(img_cropped[y_top:y_top+window,x_left:x_left+window], (64,64))
            
            # retrieve spatial features if enabled
            if spatial_bool:
                spatial_features = bin_spatial_feature_extract(img_patch,size=spatial_size)
                
            # retrieve histgoram features if enabled
            if hist_bool:
                hist_features = color_hist_feature_extract(img_patch,n_bins=n_bins)
            
            # retrieve hog features if enabled
            if hog_bool:
                if hog_channel == 0:
                    hog_features = ch1_hog[y_loc:y_loc+n_blocks_per_window,x_loc:x_loc+n_blocks_per_window].ravel()
                if hog_channel == 1:
                    hog_features = ch2_hog[y_loc:y_loc+n_blocks_per_window,x_loc:x_loc+n_blocks_per_window].ravel()
                if hog_channel == 2:
                    hog_features = ch3_hog[y_loc:y_loc+n_blocks_per_window,x_loc:x_loc+n_blocks_per_window].ravel()
                else:
                    hog_features1 =  ch1_hog[y_loc:y_loc+n_blocks_per_window,x_loc:x_loc+n_blocks_per_window].ravel()
                    hog_features2 =  ch2_hog[y_loc:y_loc+n_blocks_per_window,x_loc:x_loc+n_blocks_per_window].ravel()
                    hog_features3 =  ch3_hog[y_loc:y_loc+n_blocks_per_window,x_loc:x_loc+n_blocks_per_window].ravel()
                    hog_features = np.hstack((hog_features1,hog_features2,hog_features3))
            
            # combine all features into a single array
            X = np.hstack((spatial_features,hist_features,hog_features)).reshape(1,-1)
            
            test_features = scaler.transform(X)
            test_prediciton = svc.predict(test_features)
            
            # if prediction is true (or == 1), X is a vehicle; otherwise, it's a non-vehicle
            if test_prediciton:
                box_x_left = np.int(x_left*scale)
                box_y_top = np.int(y_top*scale)
                box_window = np.int(window*scale)

                on_windows.append(((box_x_left,box_y_top+y_start),(box_x_left+box_window,box_y_top+box_window+y_start)))
            
    return on_windows

def scaled_detection(img,svc,scaler,search_window_scales=[1,1.5,2]):
    on_windows_list = []
    y_start_stop = [400, 720] # Min and max in y to search in slide_window()
    y_starts = [y_start_stop[0], y_start_stop[0], y_start_stop[0],y_start_stop[0]]
    y_stops = [y_start_stop[1]-150,y_start_stop[1]-75,y_start_stop[1],y_start_stop[1]]
#     search_window_scales = [0.9,1.3,1.6,2]  # (64x64), (96x96), (128x128)
#     search_window_scales = [0.9,1.3,1.7,2.1]  # (64x64), (96x96), (128x128)
#     search_window_scales = [1,1.5,2]  # (64x64), (96x96), (128x128)
    for search_window_scale,y_start,y_stop in zip(search_window_scales,y_starts,y_stops):
        on_windows_temp = find_vehicles(img,y_start,y_stop,search_window_scale,svc,scaler)
        on_windows_list.extend(on_windows_temp)
    return on_windows_list

def draw_bboxes_initial(img,bboxes,color=(0,0,255),thick=2):
    color_list = [(70,130,180),(0,0,255),(0,191,255),(240,255,255),(205,92,92),(221,160,221)]
    img_copy = np.copy(img)
    # img_copy_weighted = np.copy(img)
    for bbox in bboxes:
        color = color_list[np.random.randint(0,6)]
        cv2.rectangle(img_copy,bbox[0],bbox[1],color,thick)
        # img_copy_weighted = cv2.addWeighted(img,0.4,img_copy,0.6,0) # show bounding boxes with a 0.6 alpha
    return img_copy

def generate_heatmap(img,windows_list):
    heatmap = np.zeros_like(img[:,:,0]).astype(np.uint8) # creates an array of zeros in the shape of the image
    
    # add heat to heatmap
    for window in windows_list:
        heatmap[window[0][1]:window[1][1],window[0][0]:window[1][0]] += 1
    
    # clip heatmap 
#     max_heat = 50 # limits the heat count in a region to 50
#     heatmap = np.clip(heatmap,0,max_heat)
    
    return heatmap

def draw_bboxes_final(img,heatmap_view,heatmap_buffer,n_buffer=5,threshold_limit=2,window_size_limit=1500,color=(0,0,255),thick=3,bbox_return_only=False):
    img_bboxed = np.copy(img)
    
    heatmap_buffer.append(heatmap_view)
    
    # only retains info from the last 4 frames
    if len(heatmap_buffer) > n_buffer:
        heatmap_buffer.pop(0)
    
    # define buffer weights such that the more recent frames have a larger weight
    buffer_weights = np.arange(1,len(heatmap_buffer)+1)/sum(np.arange(1,len(heatmap_buffer)+1))
    
    # apply weights to heatmaps in memory
    for b,w,i in zip(heatmap_buffer,buffer_weights,range(n_buffer)):
        heatmap_buffer[i] = b*w
    
    # filter out any values that are less than or equal to the threshold limit
    heatmap_final = np.sum(np.array(heatmap_buffer),axis=0)
    heatmap_final[heatmap_final <= threshold_limit] = 0
    
    # label() method segregates each blob within the heatmap
    labels = label(heatmap_final)
    bboxes_final = []
    for detected_vehicle in range(1,labels[1]+1):
        non_zero = (labels[0] == detected_vehicle).nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])
        
        bbox_temp = ((np.min(non_zero_x),np.min(non_zero_y)),(np.max(non_zero_x),np.max(non_zero_y)))

        x1,y1 = bbox_temp[0][0],bbox_temp[0][1]
        x2,y2 = bbox_temp[1][0],bbox_temp[1][1]
        height = y2-y1
        width = x2-x1
        area = height*width

        # window size limit is used to minimize the number of (presumably small bounding box) false postive vehicle results
        if area > window_size_limit and height > 30 and width > 30:
            bboxes_final.append(bbox_temp)
    
    # if true, return bboxes_final only
    if bbox_return_only:
        return bboxes_final
    else:
        # draws the finalized bounding boxes onto the original image
        for i,bbox in enumerate(bboxes_final):
            cv2.rectangle(img_bboxed,bbox[0],bbox[1],(0,0,255),thick)
        return img_bboxed,heatmap_final,bboxes_final

def vehicle_pipeline(img,svc,scaler,heatmap_buffer,n_buffer=1,threshold_limit=2,window_size_limit=1500,search_window_scales=[1,1.5],bbox_return_only=False):
    img_copy = np.copy(img)
    bboxes_initial = scaled_detection(img,svc,scaler,search_window_scales)
    heatmap_initial = generate_heatmap(img,bboxes_initial)
    
    if bbox_return_only:
        bboxes_final = draw_bboxes_final(img_copy,heatmap_view=heatmap_initial,heatmap_buffer=heatmap_buffer,n_buffer=n_buffer,threshold_limit=threshold_limit,window_size_limit=window_size_limit,bbox_return_only=bbox_return_only)
        return bboxes_final
    else:
        img_vehicle_bboxed,heatmap_final,bboxes_final = draw_bboxes_final(img_copy,heatmap_view=heatmap_initial,heatmap_buffer=heatmap_buffer,n_buffer=n_buffer,threshold_limit=threshold_limit,window_size_limit=window_size_limit,bbox_return_only=bbox_return_only)
        return img_vehicle_bboxed,bboxes_initial,bboxes_final,heatmap_initial,heatmap_final