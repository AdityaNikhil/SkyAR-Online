import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse
from networks import *
from skyboxengine import *
import utils
import torch
import streamlit as st
import json
from IPython.core.display import display
import imageio
from pathlib import Path
# import wandb


st.set_option('deprecation.showfileUploaderEncoding', False)

'''
# Welcome to SkyMagic!
**Developed by Aditya Digala.**

A vision-based method for video sky replacement and harmonization, which can automatically 
generate realistic and dramatic sky backgrounds in videos with controllable styles. Different from previous 
sky editing methods that either focus on static photos or require inertial measurement units integrated in smartphones on shooting videos,
 our method is purely vision-based, without any requirements on the capturing devices, and can 
 be well applied to either online or offline processing scenarios. Our method runs in real-time and is free of user interactions. 

 [Checkout wandb log of this project here](https://wandb.ai/aditya-digala-gmail-com/SkyAR-Streamlit)

**Hit Run after configuring!**
'''



def video_selector(folder_path=r'./test_videos'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a video file', filenames)
    return os.path.join(folder_path, selected_filename)


st.sidebar.header('User Input Features')
video_file = str(video_selector())
gif_file = st.sidebar.selectbox("Select a skybox",("cloudy.jpg", "jupiter.jpg", "sunset.jpg","sunny.jpg", "floatingcastle.jpg"\
                                                    ,"rainy.jpg","district9ship.jpg","supermoon.jpg","thunderstorm.mp4"))

skybox_cernter_crop = st.sidebar.slider('skybox_center_crop', 0.0, 1.0, value=0.5, step=0.1)
auto_light_matching = st.sidebar.selectbox("auto_light_matching",("True", "False"))
relighting_factor = st.sidebar.slider('relighting_factor', 0.0, 1.0, value=0.6, step=0.1)
recoloring_factor = st.sidebar.slider('recoloring_factor', 0.0, 1.0, value=0.8, step=0.1)
halo_effect = st.sidebar.selectbox("halo_effect",("True", "False"))

# parser = argparse.ArgumentParser(description='SKYAR')
# args = utils.parse_config(path_to_json="./config/config-annarbor-castle.json")

# args.ckptdir = "./checkpoints_G_coord_resnet50"

# args.datadir = video_file # choose a foreground video
# args.skybox = gif_file # choose a skybox template




# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define some helper functions for downloading pretrained model
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)




parser = argparse.ArgumentParser(description='SKYAR')
args = utils.parse_config(path_to_json="./config/config-annarbor-castle.json")

args.ckptdir = "./checkpoints_G_coord_resnet50"
args.datadir = video_file
args.skybox = gif_file
args.in_size_w = 384 # input size to sky matting model
args.in_size_h = 384 # ...
args.out_size_w = 845 # output video resolution
args.out_size_h = 480 # ...

args.skybox_center_crop = skybox_cernter_crop # view of the virtual camera
args.auto_light_matching = auto_light_matching 
args.relighting_factor = relighting_factor
args.recoloring_factor = recoloring_factor
args.halo_effect = halo_effect 



class SkyFilter():

    def __init__(self, args):

        self.ckptdir = args.ckptdir
        self.datadir = args.datadir
        self.input_mode = args.input_mode

        self.in_size_w, self.in_size_h = args.in_size_w, args.in_size_h
        self.out_size_w, self.out_size_h = args.out_size_w, args.out_size_h

        self.skyboxengine = SkyBox(args)

        self.net_G = define_G(input_nc=3, output_nc=1, ngf=64, netG=args.net_G).to(device)
        self.load_model()

        self.video_writer = cv2.VideoWriter('demo.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                                            20.0, (args.out_size_w, args.out_size_h))
        self.video_writer_cat = cv2.VideoWriter('demo-cat.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                                            20.0, (2*args.out_size_w, args.out_size_h))

        if os.path.exists(args.output_dir) is False:
            os.mkdir(args.output_dir)

        self.output_img_list = []
        self.images_list = []
        self.replaced_sky_list = []

        self.save_jpgs = args.save_jpgs

#     def load_model(self):
#         # load pretrained sky matting model
#         print('loading the best checkpoint...')
#         checkpoint = torch.load(os.path.join(self.ckptdir, 'best_ckpt.pt'))
#         self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
#         self.net_G.to(device)
#         self.net_G.eval()
            
    def load_model(self):
        cloud_model_location = "1waCarAttTQ61KFvVv2So08NneWdpwFbp"
        save_dest = Path('model')
        save_dest.mkdir(exist_ok=True)

        f_checkpoint = Path("best_ckpt.pt")

        if not f_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                from GD_download import download_file_from_google_drive
                download_file_from_google_drive(cloud_model_location, f_checkpoint)

        checkpoint = torch.load(f_checkpoint, map_location=device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()
        
        # To free memory!
        del f_checkpoint
        del checkpoint
        
#         return model        


    def write_video(self, img_HD, syneth):

        frame = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)
        self.video_writer.write(frame)

        frame_cat = np.concatenate([img_HD, syneth], axis=1)
        frame_cat = np.array(255.0 * frame_cat[:, :, ::-1], dtype=np.uint8)
        self.video_writer_cat.write(frame_cat)

        # define a result buffer
        self.output_img_list.append(frame_cat)
        img_HD = np.array(255.0 * img_HD[:, :], dtype=np.uint8)
        self.images_list.append(img_HD)
        self.replaced_sky_list.append(frame[:,:,::-1])
        

    def synthesize(self, img_HD, img_HD_prev):

        h, w, c = img_HD.shape

        img = cv2.resize(img_HD, (self.in_size_w, self.in_size_h))

        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

        with torch.no_grad():
            G_pred = self.net_G(img.to(
            ))
            G_pred = torch.nn.functional.interpolate(G_pred, (h, w), mode='bicubic', align_corners=False)
            G_pred = G_pred[0, :].permute([1, 2, 0])
            G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)
            G_pred = np.array(G_pred.detach().cpu())
            G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

        skymask = self.skyboxengine.skymask_refinement(G_pred, img_HD)

        syneth = self.skyboxengine.skyblend(img_HD, img_HD_prev, skymask)

        return syneth, G_pred, skymask

    def cvtcolor_and_resize(self, img_HD):

        img_HD = cv2.cvtColor(img_HD, cv2.COLOR_BGR2RGB)
        img_HD = np.array(img_HD / 255., dtype=np.float32)
        img_HD = cv2.resize(img_HD, (self.out_size_w, self.out_size_h))

        return img_HD
        

    def process_video(self):

        # process the video frame-by-frame

        cap = cv2.VideoCapture(self.datadir)
        m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_HD_prev = None
        
        for idx in range(m_frames):
            ret, frame = cap.read()
            if ret:
                img_HD = self.cvtcolor_and_resize(frame)

                if img_HD_prev is None:
                    img_HD_prev = img_HD

                syneth, G_pred, skymask = self.synthesize(img_HD, img_HD_prev)

                self.write_video(img_HD, syneth)

                img_HD_prev = img_HD

                if idx % 50 == 1:
                    print('processing video, frame %d / %d ... ' % (idx, m_frames))

            else:  # if reach the last frame
                break

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import streamlit as st

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

output = st.empty()
if st.button('Run'):
    # wandb.init(project="SkyAR-Streamlit")  
    if video_file and gif_file:
        with st_capture(output.code):
            sf = SkyFilter(args)
            sf.process_video()

        with imageio.get_writer('demo-img.gif', mode='I') as writer:
            for img in sf.images_list[7:47]:
                writer.append_data(img)

        with imageio.get_writer('demo-sky.gif', mode='I') as writer:
            for img in sf.replaced_sky_list[7:47]:
                writer.append_data(img) 

        st.image('demo-sky.gif', caption='Output',
           use_column_width=True)

        # wandb.log({"video": [wandb.Video('demo-sky.gif', fps=4, format="gif"),
        #          wandb.Video('demo-img.gif', fps=4, format="gif")]})







