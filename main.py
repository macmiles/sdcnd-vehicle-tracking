from moviepy.editor import VideoFileClip
from IPython.display import HTML
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lane_detect
import vehicle_detect
import cv2,glob,pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

pickle_file = 'clf_and_data.p' # pickle file used to save/load classifier
demo = False # if True, test the pipeline on an image instead of the project video
pickle_load = True # if True, load the classifier from the saved pickle file
bbox_return_only = True # if True, vehicle pipeline only returns bboxes (much1 more efficient + good for debugging purposes)

def lane_and_vehicle_pipeline(img):
    # print('lane_and_vehicle_pipeline',img.dtype)
    # retrieve lane line overlay and info
    img_lane_detect = lane_detect.lane_pipeline(img)

    # lane_pipeline undistorts the original image so if we want both lane and vehicle detections to align, we need to undistort the vehicle image as well
    img_undistorted = lane_detect.undistort_image(img)
    
    # retrieve vehicle tracking data
    if bbox_return_only:
        bboxes_final = vehicle_detect.vehicle_pipeline(img_undistorted,svc,scaler,heatmap_buffer,n_buffer=5,threshold_limit=1,window_size_limit=1500,search_window_scales=[1,1.5,2],bbox_return_only=bbox_return_only)

        img_lane_and_vehicle = np.copy(img_lane_detect)
        # apply vehicle tracking boxes to lane detected image
        for bbox in bboxes_final:
            cv2.rectangle(img_lane_and_vehicle,bbox[0],bbox[1],(0,0,255),3)
    else:
        img_vehicle_bboxed,bboxes_initial,bboxes_final,heatmap_initial,heatmap_final = vehicle_detect.vehicle_pipeline(img_undistorted,svc,scaler,heatmap_buffer,bbox_return_only=bbox_return_only)
        
        # combine lane line and vehicle tracking
        img_lane_and_vehicle = cv2.addWeighted(img_lane_detect,0.5,img_vehicle_bboxed,0.5,0)

        # displays first phase detection boxes
        img_vehicle_bboxed_initial = vehicle_detect.draw_bboxes_initial(img,bboxes_initial)

    return img_lane_and_vehicle

if __name__ == '__main__':
    # train a classifier if pickle_load is False; otherwise load classifier from pickle file
    if pickle_load == False:
        print('Loading training data.')
        # load vehicle and non-vehicle data
        vehicles = glob.glob('./vehicles/**/*.png')
        non_vehicles = glob.glob('./non-vehicles/**/*.png')

        # balance training data classes
        data_limit = min(len(vehicles),len(non_vehicles))
        vehicles = vehicles[:data_limit]
        non_vehicles = non_vehicles[:data_limit]

        print('Extracting features.')
        # extract features from images
        vehicle_features = vehicle_detect.extract_all_features(vehicles,color_space='YCrCb',hog_channel='ALL')
        non_vehicle_features = vehicle_detect.extract_all_features(non_vehicles,color_space='YCrCb',hog_channel='ALL')

        # create a single array with vehicle and non-vehicle feature data
        X = np.vstack((vehicle_features,non_vehicle_features)).astype(np.float64)

        # define y label set
        y = np.concatenate((np.ones(len(vehicle_features)),np.zeros(len(non_vehicle_features))))

        print('Normalizing training data.')
        # create a StandardScaler() object
        scaler = StandardScaler()

        # fit object to X data -- this step calculates mean and std dev for each row of data
        X_scaled = scaler.fit(X)

        # use transform method to normalize data
        X_scaled = scaler.transform(X)

        # split training and test data
        X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2)

        print('Training SVM classifier.')
        # train a SVM classifier
        svc = LinearSVC()
        svc.fit(X_train,y_train)

        print('Saving classifier.')
        # save classifier using pickle
        clf_dict = {}
        clf_dict['svc'] = svc
        clf_dict['X'] = X
        clf_dict['y'] = y
        pickle.dump(clf_dict,open(pickle_file,'wb'))
    else:
        print('Loading saved classifier.')
        # load pickle data
        with open(pickle_file,'rb') as f:
            data = pickle.load(f)
        svc = data['svc']
        X = data['X']
        y = data['y']
        
        # create a StandardScaler() object
        scaler = StandardScaler()

        # fit object to X data -- this step calculates mean and std dev for each row of data
        X_scaled = scaler.fit(X)

        # use transform method to normalize data
        X_scaled = scaler.transform(X)

    # test the pipeline on a single image by setting demo = True or 1; otherwise, run pipeline on video
    if demo:
        heatmap_buffer = []
        # img_test = mpimg.imread('./test_images/test3.jpg')
        img_test = mpimg.imread('./test_images/test4.jpg')
        # img_test = vehicle_detect.read_image('./selims_test_images/vlcsnap-2018-03-10-13h09m31s542.png')
        lane_img = lane_and_vehicle_pipeline(img_test)
        plt.figure(figsize=(20,20))
        plt.imshow(lane_img)
        plt.show()
    else:
        heatmap_buffer = []
        output_video_title = 'project_video_short_out4.mp4'
        clip1 = VideoFileClip('project_video_short.mp4')
        output_video = clip1.fl_image(lane_and_vehicle_pipeline)
        output_video.write_videofile(output_video_title, audio=False)