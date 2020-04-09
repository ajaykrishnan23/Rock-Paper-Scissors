import cv2
import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import random

class Conv2(nn.Module):
  def __init__(self):
    super(Conv2,self).__init__()
    
    self.cnn = nn.Sequential(
        nn.Conv2d(3,4,5),
        nn.Conv2d(4,5,5),
        nn.AdaptiveAvgPool2d((30,30)),     
    )
    self.lin1 = nn.Linear(4500,1000)
    self.bn = nn.BatchNorm1d(1000)
    self.act = nn.ReLU()
    self.lin2 = nn.Linear(1000,50)
    self.bn2 = nn.BatchNorm1d(50)
    self.act = nn.ReLU()
    self.do = nn.Dropout(0.5)
    self.lin3 = nn.Linear(50,3)
    self.sf = nn.Softmax()

  def forward(self,x):
    
    x = self.cnn(x)
    x = x.view(x.size(0),-1)
    x = self.lin1(x)  
    x = self.bn(x)
    x = self.act(x)
    x = self.lin2(x)
    x = self.bn2(x)
    x = self.act(x)
    #x = self.do(x)
    x = self.lin3(x)
    x = self.sf(x)
  
    return x
    
def capturevideo():
    cap = cv2.VideoCapture(0) #init video capture object arg specify which cam
    while(True):
        ret, frame = cap.read()
        cv2.rectangle(frame,(100,100),(500,500),3)
        cv2.imshow("Press \"c\" to capture the image",frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            frame = frame[100:500,100:500]
            cap.release()
            cv2.destroyAllWindows()
            break
    return frame

def calc_points(pred_text,ai_move,user,ai):
    if pred_text =='rock' and ai_move == 'paper':
        ai += 1
        return user,ai 
    elif pred_text == 'paper' and ai_move == 'rock':
        user += 1
        return user,ai
    elif pred_text == 'paper' and ai_move == 'scissors':
        ai += 1
        return user,ai
    elif pred_text == 'scissors' and ai_move == 'paper':
        user += 1
        return user,ai 
    elif pred_text == 'rock' and ai_move == 'scissors':
        user += 1
        return user, ai 
    elif pred_text == 'scissors' and ai_move == 'rock':
        ai += 1
        return user,ai

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    categories = ('rock','paper','scissors')
    model = torch.load(r'D:\Studies_extra\Rock-Paper-Scissors\FromStartLul\better_technique\resnet.pt')
    model.to(device)
    max_points = int(input("Enter maximum points"))
    user = 0
    ai = 0
    while user < max_points and ai < max_points:
        
        print("User:",user,"Ai:",ai)
        
        image = capturevideo()
        print(image.shape)
        new_array = np.transpose(image,(2,1,0))
        new_array = torch.from_numpy(new_array)
        new_array = new_array.type(torch.FloatTensor)
        logits = model(new_array.unsqueeze(0).to(device))
        print(logits)
        pred_text = categories[torch.max(logits,1)[1]]
        print("Move:",pred_text)
        
        ai_move = categories[random.randint(0,2)]
        print("AI Move:",ai_move)
        if pred_text != ai_move:
            user,ai = calc_points(pred_text,ai_move,user,ai)
        
    if user>= max_points:
        print("user wins")
    else:
        print("AI Wins")
        
main()