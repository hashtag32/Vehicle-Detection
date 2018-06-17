from moviepy.editor import VideoFileClip
import pickle

def save_training_params(clf, scaler):
    """
    Saves the parameters clf and scaler to the dictionary dict.p
    """
    dict={}
    dict["clf"]=clf
    dict["scaler"]=scaler
    pickle.dump(dict, open("dict.p","wb"))
    return True

def load_training_params():
    """
    Load the parameters that are saved into dict.p
    Returns clf and scaler
    """
    load_dict = pickle.load(open( "dict.p", "rb" ) )
    clf=load_dict["clf"]
    scaler=load_dict["scaler"]
    return clf,scaler

def write_video(Find_Carobj, inputVideo):
    """
    Function to record video
    """
    test_out_file = 'written_video.mp4'
    input_video = VideoFileClip(inputVideo)
    clip_test_out = input_video.fl_image(Find_Carobj.pipeline)
    clip_test_out.write_videofile(test_out_file, audio=False)
    return