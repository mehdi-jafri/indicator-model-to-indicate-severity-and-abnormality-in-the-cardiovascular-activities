import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from tkinter import messagebox
import sys
import os

root = tk.Tk()
root.title("")
root.geometry('1350x768+0+0')

tk.Label(root, 
		 text="CROSS TIME MULTIHEAD AND AVERAGE PERIODOGRAM BASED ABNORMALITY INDICATION FOR  ",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic").pack(pady=30,padx=0)
		 
tk.Label(root, 
		 text="ARRHYTHMIA DETECTION IN PPG SIGNALS  ",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic").pack(pady=1,padx=0)
		 
def b1():   
    
    os.system('python Plot_PPG_Signals.py')

def b2():

    os.system('python Third_Derivative_Skewness_Sliding_Window_Segmentation_Preprocessing.py')    

def b3():
    
    os.system('python Cross_Time_Residual_Multi_Head_Mutual Attention_Time_Domain_Feature_Extraction.py')
    

def b31():    

    os.system('python Average_Periodogram_Spectral_Density_Estimation_Frequency_Domain_Feature_Extraction.py')

def b32():
    os.system('python Softmax_Activation_classifier_arrhythmia_detection.py')

def b4():
    
    os.system('python MAE_Outlier_Detection.py')   
	
def b5():    
    
    os.system('python PRA.py')

def b6():
    
    os.system('python False_Positive_Rate.py')  


b1=Button(root,text="MIMIC-IV Waveform Database",command=b1,bg="black",fg="white",font = "Helvetica 13 bold italic")

b1.place(x=200,y=200)

b1.configure(width=65,height=1)


b2=Button(root,text="Third Derivative Skewness Sliding Window_Segmentation_Pre-processing ",command=b2,bg="black",fg="white",font = "Helvetica 13 bold italic")

b2.place(x=200,y=250)

b2.configure(width=65,height=1)


b3=Button(root,text="Cross-Time Residual Time-domain _Feature Extraction",command=b3,bg="black",fg="white",font = "Helvetica 13 bold italic")

b3.place(x=200,y=300)

b3.configure(width=65,height=1)


b31=Button(root,text="Average Periodogram Frequency-domain_Feature Extraction ",command=b31,bg="black",fg="white",font = "Helvetica 13 bold italic")

b31.place(x=200,y=350)

b31.configure(width=65,height=1)

b32=Button(root,text="Softmax Activation classifier_arrhythmia detection",command=b32,bg="black",fg="white",font = "Helvetica 13 bold italic")

b32.place(x=200,y=400)

b32.configure(width=65,height=1)



l2=tk.Label(root,text="Performance",fg = "light green",bg = "dark green",font = "Helvetica 16 bold italic")
l2.place(x=980,y=400)

b4=Button(root,text=" MAE of outlier detection (%)",command=b4,bg="black",fg="white",font = "Helvetica 13 bold italic")
b4.place(x=800,y=450)
b4.configure(width=45,height=1)

b5=Button(root,text="Precision, Recall and Accuracy",command=b5,bg="black",fg="white",font = "Helvetica 13 bold italic")
b5.place(x=800,y=500)
b5.configure(width=45,height=1)

b6=Button(root,text="False Positive Rate",command=b6,bg="black",fg="white",font = "Helvetica 13 bold italic")
b6.place(x=800,y=550)
b6.configure(width=45,height=1)

root.mainloop()
