import os
import cv2
import tkinter as tk
from tkinter import ttk
import PIL.Image, PIL.ImageTk, PIL.ImageGrab
from matplotlib import pyplot as plt
import numpy as np
import mediapipe as mp
import math
import time
import csv
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from config import FILE_PATH

global CAMERA_SHOW
CAMERA_SHOW = 0

global COUNT_STIMULUS
COUNT_STIMULUS = 1

global FRAME_COUNT
FRAME_COUNT = 0

global DOT_ROW
DOT_ROW = 0

global DOT_COLUMN
DOT_COLUMN = 0

global DOT_NUM
DOT_NUM = 1

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
NOSE = [168, 4]

mp_face_mesh = mp.solutions.face_mesh

def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def center_point(point_of_left_eye, point_of_right_eye):
    x1, y1 = point_of_left_eye.ravel()
    x2, y2 = point_of_right_eye.ravel()

    xC = (x1+x2)/2
    yC = (y1+y2)/2

    return xC, yC

def make_directory(username):
    path = os.path.join(FILE_PATH, username)
    os.makedirs(path, exist_ok=True)
    filename = f'{path}/{username}_calibration_collected_data.csv'
    columns = ['Vector x', 'Vector y', 'Dot center x', 'Dot center y']
    with open(filename, 'a+', newline='', encoding='UTF8') as csvfile_calibration:
        csvwriter = csv.writer(csvfile_calibration)
        csvwriter.writerow(columns)

def write_calibration_data(username, vector_x, vector_y, dot_center_x, dot_center_y):
    columns = ['Vector x', 'Vector y','Dot center x', 'Dot center y']
    row = [f'{vector_x}', f'{vector_y}', f'{dot_center_x}', f'{dot_center_y}']
    filename = f"{username}/{username}_calibration_collected_data.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, 'a+', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def read_calibration_data(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/{username}_calibration_collected_data.csv', delimiter=',', usecols= ['Vector x', 'Vector y','Dot center x', 'Dot center y'])
    return df

def write_stimulus_eye_gaze_data(username, vector_x, vector_y, left_eye_blink_check, right_eye_blink_check, time):
    columns = ['Vector x', 'Vector y', 'Left eye blink check','Right eye blink check', 'Time']
    row =  [f'{vector_x}', f'{vector_y}', f'{left_eye_blink_check}', f'{right_eye_blink_check}',f'{time}']
    filename = f"{username}/{username}_gaze_data_stimulus_{COUNT_STIMULUS-1}.csv" #-1
    file_exists = os.path.isfile(filename) 
    with open(filename, 'a+', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def read_eye_gaze_data(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/{username}_gaze_data_stimulus_{COUNT_STIMULUS}.csv', delimiter=',', usecols= ['Vector x', 'Vector y', 'Left eye blink check','Right eye blink check', 'Time'])
    return df.to_numpy()

def write_predictions(username, X_screen_position, Y_screen_position, time):
    columns = ['Prediction X', 'Prediction Y', 'Time in ms']
    row = [f'{X_screen_position}', f'{Y_screen_position}', f'{time}']
    filename = f'{username}/{username}_stimulus_{COUNT_STIMULUS}_predictions.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a+', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def read_predictions(username):
    path = os.path.join(FILE_PATH,username)
    df = pd.read_csv(f'{path}/{username}_stimulus_{COUNT_STIMULUS}_predictions.csv', delimiter=',', usecols= ['Prediction X', 'Prediction Y', 'Time in ms'])
    return df

def write_experiment_parameters(username, vector_x, vector_y, event_x, event_y, prediction_x, prediction_y):
    columns = ['Vector x', 'Vector y', 'Event x', 'Event y', 'Prediction x', 'Prediction y']
    row = [f'{vector_x}', f'{vector_y}', f'{event_x}', f'{event_y}', f'{prediction_x}', f'{prediction_y}']
    filename = f'{username}/{username}_experiment_parameters.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, '+a', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def read_experiment_parameters(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/{username}_experiment_parameters.csv', delimiter=',', usecols=['Vector x', 'Vector y', 'Event x', 'Event y', 'Prediction x', 'Prediction y'])
    return df

def write_additional_data(username, real_x, real_y, prediction_x, prediction_y, vector_x, vector_y):
    columns = ['Real x', 'Real y', 'Prediction x', 'Prediction y', 'Vector x', 'Vector y']
    row = [f'{real_x}', f'{real_y}', f'{prediction_x}', f'{prediction_y}', f'{vector_x}', f'{vector_y}']
    filename = f'{username}/{username}_collected_training_data.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, '+a', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def read_additional_data(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/{username}_collected_training_data.csv', delimiter=',', usecols=['Real x', 'Real y', 'Prediction x', 'Prediction y', 'Vector x', 'Vector y'])
    return df

def read_data_for_evaluation(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/random_dots_for_evaluation.csv', delimiter=',', usecols=['Real x', 'Real y', 'Prediction x', 'Prediction y', 'Vector x', 'Vector y' ])
    return df

def calculate_gaze_position(slope, intercept, x):
    return slope*x + intercept

class App(tk.Tk): 
    def __init__(self):
        super().__init__()

        self.title('Eye tracking app')
        
        self.iconphoto(False, tk.PhotoImage(file='eye_track_icon.png'))
        self.attributes('-fullscreen', True)

        container = tk.Frame(self, bg="#FFFFFF", height=self.winfo_screenheight(), width=self.winfo_screenwidth())
       
        container.pack(fill="both", expand = True)
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side=tk.BOTTOM)
        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.canvas.pack(pady=120)
     
        self.label_username = tk.Label(self.bottom_frame, text="Username:")
        self.label_username.pack(side=tk.LEFT)
        self.entry_username = tk.Entry(self.bottom_frame)
        self.entry_username.pack(padx=5, side=tk.LEFT)
        self.btn_save_username = ttk.Button(self.bottom_frame, text='Save', command=self.get_input)
        self.btn_save_username.pack(padx=5, pady=5, side=tk.BOTTOM)
        self.open_camera_check(CAMERA_SHOW) 
        
    def open_camera_check(self, video_source):
        self.video = MyVideoCapture(CAMERA_SHOW)
        self.canvas.configure(width=self.video.width, height=self.video.height)
        self.canvas.pack()

        self.delay = 12
        # if CAMERA_SHOW == 0:
        self.mode = CAMERA_SHOW
        self.update()

    def get_input(self):
        print('Name: ', self.entry_username.get()) 
        user = self.entry_username.get()
        if(user != ""): 
            self.username = user
            self.bottom_frame.destroy()
            self.button_next = ttk.Button(self, text='Next', command=self.calibrate)
            self.button_next.pack(side=tk.RIGHT, padx=5, pady=5)
            make_directory(user)
        else:
            self.msg_username_empty = tk.Label(self.bottom_frame, text="Username required")
            self.msg_username_empty.pack(side=tk.BOTTOM)

    def update(self):
        global FRAME_COUNT
        ret, rgb_frame = self.video.get_frame()

        if ret and self.mode==0:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(rgb_frame))
            self.image_on_canvas = self.canvas.create_image(0,0, image=self.photo, anchor=tk.NW)

        FRAME_COUNT += 1
        
        if CAMERA_SHOW==2 or CAMERA_SHOW==3 or CAMERA_SHOW==5:
            
            current_username = self.username
            if current_username:
                current_vector_x = self.video.vector_x
                current_vector_y = self.video.vector_y
                current_left_eye_blink_check = self.video.left_eye_blink_check
                current_right_eye_blink_check = self.video.right_eye_blink_check

                if CAMERA_SHOW == 3:
                    t = time.time()
                    current_time_ms = int(t*1000)
                    write_stimulus_eye_gaze_data(current_username, current_vector_x, current_vector_y, current_left_eye_blink_check, current_right_eye_blink_check, current_time_ms)
        
        self.after(self.delay, self.update)

    def calibrate(self):
        global CAMERA_SHOW
        CAMERA_SHOW = 1
        self.mode = CAMERA_SHOW
        ret, frame = self.video.get_frame()
        self.canvas.delete('all')      
        self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth(), bg="#494949")
        self.canvas.pack(fill="both", expand=True, pady=0, padx=0)
        self.draw_dot()
        
    def dot_click_event(self, N, dot_center_x, dot_center_y, event):
        global DOT_COLUMN
        global DOT_ROW
        global DOT_NUM
       
        closest = self.canvas.find_closest(event.x, event.y)
        self.canvas.delete(closest)
                
        # current_frame = self.study_frame()

        current_username = self.username
        if current_username:
            
            current_vector_x = self.video.vector_x
            current_vector_y = self.video.vector_y
            write_calibration_data(current_username, current_vector_x, current_vector_y, dot_center_x, dot_center_y)
            # rgb_frame_photo = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("calculating_parameters-" + f'{N}' + ".jpg", rgb_frame_photo)

        if DOT_NUM < 9:
            DOT_NUM += 1
            self.draw_dot()
        else :
            print("Last dot.")
            self.get_multi_poly_regression_parameters()
            self.collect_training_data()

    def draw_dot(self):

        global DOT_NUM
        window_width = self.winfo_screenwidth()
        window_height = self.winfo_screenheight()

        dot_radius = 13 
       
        origin_x = 0
        origin_y = 0
        
        if DOT_NUM == 1:
            x0 = origin_x
            y0 = origin_y
            x1 = origin_x+2*dot_radius
            y1 = origin_y+2*dot_radius
        elif DOT_NUM ==2:
            x0 = window_width/4-dot_radius
            y0 = window_height/4-dot_radius
            x1 = window_width/4+dot_radius
            y1 = window_height/4+dot_radius
        elif DOT_NUM == 3:
            x0 = window_width*3/4-dot_radius
            y0 = window_height/4-dot_radius
            x1 = window_width*3/4+dot_radius
            y1 = window_height/4+dot_radius
        elif DOT_NUM == 4:
            x0 = window_width-2*dot_radius
            y0 = origin_y
            x1 = window_width
            y1 = origin_y+2*dot_radius
        elif DOT_NUM == 5:
            x0 = origin_x
            y0 = window_height-2*dot_radius
            x1 = origin_x+2*dot_radius
            y1 = window_height
        elif DOT_NUM == 6:
            x0 = window_width/4-dot_radius
            y0 = window_height*3/4-dot_radius
            x1 = window_width/4+dot_radius
            y1 = window_height*3/4+dot_radius
        elif DOT_NUM == 7:
            x0 = window_width/2-dot_radius
            y0 = window_height/2-dot_radius
            x1 = window_width/2+dot_radius
            y1 = window_height/2+dot_radius
        elif DOT_NUM == 8:
            x0 = window_width*3/4-dot_radius
            y0 = window_height*3/4-dot_radius
            x1 = window_width*3/4+dot_radius
            y1 = window_height*3/4+dot_radius
        elif DOT_NUM == 9:
            x0 = window_width-2*dot_radius
            y0 = window_height-2*dot_radius
            x1 = window_width
            y1 = window_height

        globals()[f'attention_dot_{DOT_NUM}'] = self.canvas.create_oval(x0, y0, x1, y1, fill='#DE4C4C', outline='#DE4C4C')
        dot_center_x = x0 + dot_radius
        dot_center_y = y0 + dot_radius
        self.canvas.pack()
        self.canvas.tag_bind(globals()[f'attention_dot_{DOT_NUM}'], '<Button-1>', lambda event, dot_num = DOT_NUM, dot_center_x = dot_center_x, dot_center_y = dot_center_y: self.dot_click_event(dot_num, dot_center_x, dot_center_y, event))

    def stimulus_attention_tracking(self):
        print("Stimulus")
        global CAMERA_SHOW

        CAMERA_SHOW = 3
        self.mode = CAMERA_SHOW

        self.canvas.delete('all')      
        self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth())
        self.canvas.pack(fill="both", expand=True, pady=0, padx=0)
        self.msg_flag = 1
        self.display_message('Observe following 3 pictures')
        
    def next_stimulus(self):
        global COUNT_STIMULUS
    
        if COUNT_STIMULUS == 4:     
            # self.msg_flag = 2
            # self.canvas.delete('all')
            # self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth(), bg="#494949")
            # self.canvas.pack(fill="both", expand=True, pady=0, padx=0)
            # self.display_message('Follow the dot across the screen') 
            self.save_results()

        else:
            image = PIL.Image.open(f"visual_stimuli/stimulus{COUNT_STIMULUS}.jpg")
            self.stimulus_image_width = self.canvas.winfo_screenwidth()
            self.stimulus_image_height = self.canvas.winfo_screenheight()
            image = image.resize((self.stimulus_image_width, self.stimulus_image_height), PIL.Image.Resampling.LANCZOS)
            self.current_stimulus = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(self.winfo_screenwidth()/2, self.winfo_screenheight()/2, image=self.current_stimulus, anchor=tk.CENTER)
            self.canvas.pack()
            COUNT_STIMULUS += 1
     
            self.after(10000, self.next_stimulus)

    def support_vector_regression(self, X_values, Y_values):

        training_data = read_additional_data(self.username)

        X = training_data.loc[:, f'{X_values}']
        y = training_data.loc[:, f'{Y_values}']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        svr = SVR(kernel='linear', C=1.0)
        svr.fit(X_train.to_numpy().reshape(-1,1), y_train.ravel())
        y_pred = svr.predict(X_test.to_numpy().reshape(-1,1))
        
        r2 = r2_score(y_test.ravel(), y_pred.ravel())
        new_slope, new_intercept, r_value, p_value, std_err = stats.linregress(y_pred.ravel(), y_test.ravel()) #obrni
        mse = mean_squared_error(y_test.ravel(), y_pred.ravel())
        rmse = math.sqrt(mse)
       
        file = open(f"{FILE_PATH}/{self.username}/metrics.txt", "a")
        file.write(f"R2 score for {X_values}: {r2}" + "\n")
        file.write(f"MSE in svr (for {X_values} and {Y_values} relationship): {mse}" + "\n")
        file.write(f"RMSE in svr (for {X_values} and {Y_values} relationship): {rmse}" + "\n")
        file.close()

        return new_slope, new_intercept
    
    def get_multi_poly_regression_prediction(self):
        
        if self.username:
            
            current_vector_x = self.video.vector_x
            current_vector_y = self.video.vector_y
            current_left_eye_blink_check = self.video.left_eye_blink_check
            current_right_eye_blink_check = self.video.right_eye_blink_check

            predicted_X = self.a_coefficients[0] + self.a_coefficients[1]*current_vector_x + self.a_coefficients[2]*current_vector_y + self.a_coefficients[3]*current_vector_x*current_vector_y + self.a_coefficients[4]*pow(current_vector_x,2) + self.a_coefficients[5]*pow(current_vector_y,2)
            predicted_Y = self.b_coefficients[0] + self.b_coefficients[1]*current_vector_x + self.b_coefficients[2]*current_vector_y + self.b_coefficients[3]*current_vector_x*current_vector_y + self.b_coefficients[4]*pow(current_vector_x,2) + self.b_coefficients[5]*pow(current_vector_y,2)
           
            return predicted_X, predicted_Y, current_vector_x, current_vector_y, current_left_eye_blink_check, current_right_eye_blink_check
        
    def experiment1_calculate_error(self):  

        global CAMERA_SHOW
        CAMERA_SHOW = 5
        self.mode = CAMERA_SHOW

        self.canvas.delete('all')
        self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth(), bg="#555555")
        self.canvas.pack(fill="both", expand=True, pady=0, padx=0)

        self.calculate_cost_function_after()

    def collect_training_data(self):

        global CAMERA_SHOW
        CAMERA_SHOW = 6
        self.mode = CAMERA_SHOW

        self.canvas.delete('all')
        self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth(), bg="#494949")
        self.canvas.pack(fill="both", expand=True, pady=0, padx=0)

        self.messages = ["Follow the dot across the screen", "3", "2", "1"]
        self.msg_flag = 0
        self.message_index = 0
        self.display_messages()
    
    def display_messages(self):
        if self.message_index < len(self.messages):
            message = self.messages[self.message_index]
            print("Index: ", self.message_index)
            self.display_message(message)
            self.message_index += 1
            self.after(1200, self.display_messages) 
            
    def remove_message(self):
        self.canvas.delete(self.msgId)
        if self.message_index == 3:
                self.after(1200,self.draw_animation)  
        if self.msg_flag == 1:
            self.next_stimulus()
        if self.msg_flag == 2:
            self.experiment1_calculate_error()    
  
    def display_message(self, message):
        self.msgId = self.canvas.create_text(self.winfo_width()/2, self.winfo_height()/2, text=message, font=('Arial', 20), fill='white')
        self.after(1200, self.remove_message)

    def draw_animation(self):
        
        if self.message_index == 3:
            self.canvas.delete(self.msgId)
                
        dot_radius = 13
        origin_x = 0
        origin_y = 0

        x0 = origin_x-dot_radius 
        y0 = origin_y-dot_radius
        x1 = origin_x + dot_radius
        y1 = origin_y + dot_radius
        
        self.ball = self.canvas.create_oval(x0, y0, x1, y1, fill='#DE4C4C', outline='#DE4C4C')
        self.animate()  

    def calculate_cost_function(self):

        data = read_additional_data(self.username)
        random_subset = data.sample(n=15, random_state=42)
        subset_data_df = pd.DataFrame(random_subset)
        subset_data_df.to_csv(f"{self.username}/random_dots_for_evaluation.csv", index=False)
       
        predicted_values_X = subset_data_df.iloc[:,2]
        actual_values_X = subset_data_df.iloc[:,0]
        predicted_values_Y = subset_data_df.iloc[:,3] 
        actual_values_Y = subset_data_df.iloc[:,1]
        mse_X = mean_squared_error(actual_values_X, predicted_values_X)
        mse_Y = mean_squared_error(actual_values_Y, predicted_values_Y)
        Rmse_X = math.sqrt(mse_X)
        Rmse_Y = math.sqrt(mse_Y)    

        file = open(f"{FILE_PATH}/{self.username}/metrics.txt", "w")
        file.write("MSE before exp:\n")
        file.write(f"MSE for X: {mse_X}" + "\n")
        file.write(f"MSE for Y: {mse_Y}" + "\n")
        file.write(f"RMSE for X: {Rmse_X}" + "\n")
        file.write(f"RMSE for Y: {Rmse_Y}" + "\n")
        file.close()
    
    def remove_dot(self):
        self.canvas.delete(self.dot_evaluation)
        self.draw_evaluation_dot(self.df)

    def get_eye_gaze_predictions(self):
       
        predicted_X, predicted_Y, current_vector_X, current_vector_Y, current_left_eye_blink_check, current_right_eye_blink_check = self.get_multi_poly_regression_prediction()

        trained_gaze_X_prediction = calculate_gaze_position(self.new_slope_horizontal, self.new_intercept_horizontal, predicted_X)
        trained_gaze_Y_prediction = calculate_gaze_position(self.new_slope_vertical, self.new_intercept_vertical, predicted_Y)

        return current_vector_X, current_vector_Y, trained_gaze_X_prediction, trained_gaze_Y_prediction          

    def draw_evaluation_dot(self, df):
        dot_radius = 13
        self.df =df
        if self.row < len(df):
            dot_X = df.iloc[self.row, 0]
            dot_Y = df.iloc[self.row, 1]
            self.dot_evaluation = self.canvas.create_oval(dot_X-dot_radius, dot_Y-dot_radius, dot_X+dot_radius, dot_Y+dot_radius, fill='#DE4C4C', outline='#DE4C4C')
            v_x, v_y, pred_X, pred_Y = self.get_eye_gaze_predictions()
            write_experiment_parameters(self.username,v_x, v_y, dot_X, dot_Y, pred_X, pred_Y)
            self.row += 1
            self.after(1100, self.remove_dot)
        elif self.row >= len(df):
            eye_data = read_experiment_parameters(self.username)
    
            actual_values_X = eye_data.iloc[:-1, 2]
            actual_values_Y = eye_data.iloc[:-1, 3]
            predicted_values_X = eye_data.iloc[:-1, 4]
            predicted_values_Y = eye_data.iloc[:-1, 5]
            
            mse_X = mean_squared_error(actual_values_X, predicted_values_X)
            mse_Y = mean_squared_error(actual_values_Y, predicted_values_Y)
            Rmse_X = math.sqrt(mse_X)
            Rmse_Y = math.sqrt(mse_Y)

            file = open(f"{FILE_PATH}/{self.username}/metrics.txt", "a")
            file.write("MSE after experiment:\n")
            file.write(f"MSE for X: {mse_X}" + "\n")
            file.write(f"MSE for Y: {mse_Y}" + "\n")
            file.write(f"RMSE for X: {Rmse_X}" + "\n")
            file.write(f"RMSE for Y: {Rmse_Y}" + "\n")
            file.close()

    def calculate_cost_function_after(self):
        data = read_data_for_evaluation(self.username)
        self.row = 0
        self.draw_evaluation_dot(data)
   
    def animate(self):

        dot_radius = 13 
        
        x0, y0, x1, y1 = self.canvas.coords(self.ball)
        horizontal_space = self.canvas.winfo_width()/8
        vertical_space = self.canvas.winfo_height()/8
 
        self.wait_time = 1100

        predicted_X, predicted_Y, vector_x, vector_y, left_eye_blink, right_eye_blink = self.get_multi_poly_regression_prediction()

        if not ((left_eye_blink>=0 and left_eye_blink<=4) and (right_eye_blink>=0 and right_eye_blink<=4)):
            write_additional_data(self.username, x1, y1, predicted_X, predicted_Y, vector_x, vector_y) 

        if y1 < vertical_space+dot_radius:
            if x1 < horizontal_space+dot_radius:
                self.canvas.move(self.ball, horizontal_space, vertical_space) 
                self.after(self.wait_time, self.animate)
        elif y1 == vertical_space+dot_radius or y1 == 3*vertical_space+dot_radius or y1 == 5*vertical_space+dot_radius:
            if x1 < horizontal_space+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0) 
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space+dot_radius and x1 < horizontal_space*7+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0)
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space*7+dot_radius:
                self.canvas.move(self.ball, 0, vertical_space)
                self.after(self.wait_time, self.animate)
        elif y1 == 2*vertical_space+dot_radius or y1 == 4*vertical_space+dot_radius or y1 == 6*vertical_space+dot_radius:
            if x1 < horizontal_space+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0) 
                self.after(self.wait_time, self.animate)
            elif x1 <= horizontal_space+dot_radius:
                self.canvas.move(self.ball, 0, vertical_space) 
                self.after(self.wait_time, self.animate)
            elif x1 >= 2*horizontal_space+dot_radius and x1 < horizontal_space*8+dot_radius:
                self.canvas.move(self.ball, -horizontal_space, 0)
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space*7+dot_radius:
                self.canvas.move(self.ball, -horizontal_space, 0)
                self.after(self.wait_time, self.animate)
        elif y1 == 7*vertical_space+dot_radius:
            if x1 < horizontal_space+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0) 
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space+dot_radius and x1 < horizontal_space*7+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0)
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space*7+dot_radius:
                self.calculate_cost_function() 
                self.new_slope_horizontal, self.new_intercept_horizontal = self.support_vector_regression('Prediction x', 'Real x')
                self.new_slope_vertical, self.new_intercept_vertical = self.support_vector_regression('Prediction y', 'Real y')
                self.stimulus_attention_tracking()
        else:
            print("Out of screen. Exp stop.") 
            self.calculate_cost_function() 
            self.new_slope_horizontal, self.new_intercept_horizontal = self.support_vector_regression('Prediction x', 'Real x')
            self.new_slope_vertical, self.new_intercept_vertical = self.support_vector_regression('Prediction y', 'Real y')
            self.stimulus_attention_tracking()     

    def get_multi_poly_regression_parameters(self):
        data = read_calibration_data(self.username).to_numpy()
        vx = data[:, 0]
        vy = data[:, 1]
        ux = data[:, 2]
        uy = data[:, 3]

        X = np.column_stack((np.ones_like(vx), vx, vy, vx * vy, vx**2, vy**2))
        a, _, _, _ = np.linalg.lstsq(X, ux, rcond=None)
        b, _, _, _ = np.linalg.lstsq(X, uy, rcond=None)

        self.a_coefficients = a
        self.b_coefficients = b

    def save_predictions_of_stimulus_gaze(self):
        global COUNT_STIMULUS

        if self.username:
            eye_gaze_data = read_eye_gaze_data(self.username)

            for row in eye_gaze_data:
                if not((row[2]>=0 and row[2]<=4) and (row[3]>=0 and row[3]<=4)):
                       
                    current_vector_x = row[0]
                    current_vector_y = row[1]
                    gaze_time = row[4]

                    eyes_predicted_X = self.a_coefficients[0] + self.a_coefficients[1]*current_vector_x + self.a_coefficients[2]*current_vector_y + self.a_coefficients[3]*current_vector_x*current_vector_y + self.a_coefficients[4]*pow(current_vector_x,2) + self.a_coefficients[5]*pow(current_vector_y,2)
                    eyes_predicted_Y = self.b_coefficients[0] + self.b_coefficients[1]*current_vector_x + self.b_coefficients[2]*current_vector_y + self.b_coefficients[3]*current_vector_x*current_vector_y + self.b_coefficients[4]*pow(current_vector_x,2) + self.b_coefficients[5]*pow(current_vector_y,2)

                    trained_prediction_X = eyes_predicted_X*self.new_slope_horizontal + self.new_intercept_horizontal
                    trained_prediction_Y = eyes_predicted_Y*self.new_slope_vertical + self.new_intercept_vertical

                    write_predictions(self.username, trained_prediction_X, trained_prediction_Y, gaze_time)

                else:
                    print("Blink - removed!")
    
    def save_heatmap(self):
        eye_tracking_data = read_predictions(self.username)
        eye_tracking_data = eye_tracking_data.to_numpy()

        image_path = f"{FILE_PATH}/visual_stimuli/stimulus{COUNT_STIMULUS}.jpg"
        image = cv2.imread(image_path)

        canvas_width, canvas_height = (self.canvas.winfo_width(), self.canvas.winfo_height()) 

        heatmap, xedges, yedges = np.histogram2d(
            eye_tracking_data[:, 0],
            eye_tracking_data[:, 1],
            bins=80
        )

        smoothed_heatmap = gaussian_filter(heatmap, sigma=2)
        normalized_heatmap = (smoothed_heatmap - np.min(smoothed_heatmap)) / (np.max(smoothed_heatmap) - np.min(smoothed_heatmap))
        resized_image = cv2.resize(image, (canvas_width, canvas_height))
        heatmap_overlay = cv2.resize(cv2.applyColorMap(np.uint8(normalized_heatmap * 255), cv2.COLORMAP_JET), (canvas_width, canvas_height))
        heatmap_overlay_image = cv2.addWeighted(resized_image, 0.7, heatmap_overlay, 0.3, 0)
        heatmap_overlay_rgb = cv2.cvtColor(heatmap_overlay_image, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(canvas_width / 100, canvas_height / 100))
        plt.imshow(heatmap_overlay_rgb)
        plt.axis('off')
        plt.savefig(f'{FILE_PATH}/{self.username}/heatmap_stimulus{COUNT_STIMULUS}.png')
        plt.close(fig)
    
    def save_gazeplot(self):

        image_path = f'{FILE_PATH}/visual_stimuli/stimulus{COUNT_STIMULUS}.jpg'
        original_image = cv2.imread(image_path)
        resized_image = cv2.resize(original_image, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        plt.imshow(resized_image_rgb)
       
        gaze_data = read_predictions(self.username)
        X = gaze_data['Prediction X']
        Y = gaze_data['Prediction Y']
        max_x = 1536
        max_y = 864
        timestamps_ms = gaze_data['Time in ms']
        timestamps_sec = [t / 1000 for t in timestamps_ms]
        plt.scatter(X, Y, c=timestamps_sec, cmap='viridis', marker='o')
        cbar = plt.colorbar()
       
        plt.xlim(0,max_x)
        plt.ylim(0,max_y)
        plt.gca().invert_yaxis()
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        cbar.set_label('Timestamp (seconds)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title("Gaze Plot")
        
        plt.savefig(f'{FILE_PATH}/{self.username}/gaze_plot_stimulus{COUNT_STIMULUS}.png')

        plt.clf()
 
    def generate_gazeplot(self):
        counter = 1
        dot_radius = 13
        global COUNT_STIMULUS

        gaze_positions = read_predictions(self.username).to_numpy()
        image = PIL.Image.open(f"visual_stimuli/stimulus{COUNT_STIMULUS}.jpg")
        image = image.resize((self.stimulus_image_width, self.stimulus_image_height), PIL.Image.Resampling.LANCZOS)
        self.current_stimulus_img = PIL.ImageTk.PhotoImage(image=image)
        self.canvas.create_image(self.winfo_screenwidth()/2, self.winfo_screenheight()/2, image=self.current_stimulus_img, anchor=tk.CENTER)

        for row in gaze_positions:
            self.canvas.create_oval(row[0]-dot_radius, row[1]-dot_radius, row[0]+dot_radius, row[1]+dot_radius, fill='#3399FF', outline='#3399FF')
            self.canvas.create_text(row[0], row[1], text=counter)
            counter+=1 
        self.canvas.pack()

        screenshot = PIL.ImageGrab.grab()
        screenshot.save(f"{self.username}/stimulus{COUNT_STIMULUS}_gazeplot.jpg")
        self.canvas.delete('all')

    def save_results(self):
        global COUNT_STIMULUS
        COUNT_STIMULUS = 1
        stimuli_num = 3

        for stimulus in range(stimuli_num):
            self.save_predictions_of_stimulus_gaze()
            print(f"Wait...Creating diagrams for stimulus {COUNT_STIMULUS}: ")
            self.save_gazeplot()
            self.save_heatmap()
            COUNT_STIMULUS += 1

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.video = cv2.VideoCapture(video_source)
        
        if not self.video.isOpened():
             raise ValueError("Unable to open video source", video_source)
        
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                # if self.video.isOpened():
                ret, frame = self.video.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                             
                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])   
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                        
                    center_of_inner_corners_X, center_of_inner_corners_Y = center_point(mesh_points[LEFT_EYE][8], mesh_points[RIGHT_EYE][0])
                 
                    center_of_iris_centers_X, center_of_iris_centers_Y = center_point(center_left, center_right)

                    distance_corners = euclidean_distance(mesh_points[LEFT_EYE][8], mesh_points[RIGHT_EYE][0])
                    distance_nose_ends = euclidean_distance(mesh_points[NOSE][0], mesh_points[NOSE][1])
                    
                    right_eye_eyelids_distance = euclidean_distance(mesh_points[RIGHT_EYE][12], mesh_points[RIGHT_EYE][4])
                    left_eye_eyelids_distance = euclidean_distance(mesh_points[LEFT_EYE][12], mesh_points[LEFT_EYE][4])

                    if (right_eye_eyelids_distance>=0 and right_eye_eyelids_distance<=3) and (left_eye_eyelids_distance>=0 and left_eye_eyelids_distance<=3):
                        print("Blinked")

                    self.vector_x = (center_of_iris_centers_X-center_of_inner_corners_X)/distance_corners
                    self.vector_y = (center_of_iris_centers_Y-center_of_inner_corners_Y)/distance_nose_ends
                    self.left_eye_blink_check = left_eye_eyelids_distance
                    self.right_eye_blink_check = right_eye_eyelids_distance

                    #render observed landmarks
                    cv2.circle(frame, center_left, 1, (255, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, center_right, 1, (255, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[RIGHT_EYE][0], 1, (255, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[LEFT_EYE][8], 1, (255, 255, 0), -1, cv2.LINE_AA)

                    cv2.circle(frame, mesh_points[NOSE][0], 1, (255, 255, 0), -1, cv2.LINE_AA) #nose top
                    cv2.circle(frame, mesh_points[NOSE][1], 1, (255, 255, 0), -1, cv2.LINE_AA) #nose 

                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    app = App()
    app.mainloop()   
