#!/usr/bin/env python
# coding: utf-8

#from shape_learning.shape_learner_manager import ShapeLearnerManager
#from shape_learning.shape_learner import SettingsStruct
#from shape_learning.shape_modeler import ShapeModeler #for normaliseShapeHeight()

import os.path

import numpy as np
import matplotlib.pyplot as plt


#from kivy.config import Config
#Config.set('kivy', 'logger_enable', 0)
#Config.write()

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line


from scipy import interpolate

import argparse
parser = argparse.ArgumentParser(description='Learn a collection of drawings')
parser.add_argument('draw', action="store",
                help='The draw to be learnt')

class Stroke:
    def __init__(self, x=None, y=None):
        self.x = []
        self.y = []
        if x:
            self.x = x
        if y:
            self.y = y
        self.len = min(len(self.x),len(self.y))

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_len(self):
        return self.len

    def append(self,x,y):
        self.x.append(x)
        self.y.append(y)
        self.len+=1

    def downsampleShape(self, numDesiredPoints):
        """ change the length of a stroke with interpolation"""

        t_current = np.linspace(0, 1, len(self.x))
        t_desired = np.linspace(0, 1, numDesiredPoints)

        f = interpolate.interp1d(t_current, self.x, kind='linear')
        self.x = f(t_desired).tolist()
        f = interpolate.interp1d(t_current, self.y, kind='linear')
        self.y = f(t_desired).tolist()

        self.len = numDesiredPoints

    def euclidian_length(self):
        """comput length of the shape """

        shape_length = 0
        last_x = self.x
        last_y = self.y
        scale = [0]
        for i in range(self.len-1):
            x = np.array(self.x[i+1])
            y = np.array(self.y[i+1])
            last_x = np.array(self.x[i])
            last_y = np.array(self.y[i])
            shape_length += np.sqrt((x-last_x)**2 + (y-last_y)**2)
            scale.append(shape_length)
        return shape_length, scale


    def uniformize(self):
        """make the distribution of points in the stroke uniform """

        self.len = len(self.x)

        if self.len>1:
            # comput length of the shape:
            shape_length,scale = self.euclidian_length()

            # find new points:
            new_shape = Stroke()
            step = shape_length/float(self.len)
            biggest_smoller_point = 0
            new_shape.append(self.x[0], self.y[0])
            for i in 1+np.array(range(len(self.x)-1)):
                try:
                    while i*step > scale[biggest_smoller_point]:
                        biggest_smoller_point += 1

                    biggest_smoller_point -= 1
                    x0 = self.x[biggest_smoller_point]
                    y0 = self.y[biggest_smoller_point]
                    x1 = self.x[biggest_smoller_point+1]
                    y1 = self.y[biggest_smoller_point+1]
                    diff = float(i*step-scale[biggest_smoller_point])
                    dist = float(scale[biggest_smoller_point+1]-scale[biggest_smoller_point])
                    new_x = x0 + diff*(x1-x0)/dist
                    new_y = y0 + diff*(y1-y0)/dist
                    new_shape.append(new_x, new_y)

                except IndexError:
                    print i*step
                    print biggest_smoller_point
                    print scale
            new_shape.append(self.x[-1], self.y[-1])


            self.x = new_shape.x
            self.y = new_shape.y
            self.len = new_shape.len


    def revert(self):
        self.x = self.x[::-1]
        self.y = self.y[::-1]
        return self

    def get_center(self):
        x = np.array(self.x)
        y = np.array(self.y)
        return np.mean(x), np.mean(y)

    def normalize(self):
        x_min = min(self.x)
        x_max = max(self.x)
        y_min = min(self.y)
        y_max = max(self.y)

        x_range = x_max - x_min
        y_range = y_max - y_min

        x = np.array(self.x)
        y = np.array(self.y)
        x -= x_min
        y -= y_min
        x = x/float(x_range)
        y = y/float(y_range)

        self.x = x.tolist()
        self.y = y.tolist()

    def normalize_wrt_x(self):
        x_min = min(self.x)
        x_max = max(self.x)
        y_min = min(self.y)

        x_range = x_max - x_min

        x = np.array(self.x)
        y = np.array(self.y)
        x -= x_min
        y -= y_min
        x = x/float(x_range)
        y = y/float(x_range)

        self.x = x.tolist()
        self.y = y.tolist()


def concat(strokes):
    long_stroke = Stroke()
    for stroke in strokes:
        long_stroke.x += stroke.x
        long_stroke.y += stroke.y
        long_stroke.len += stroke.len
    return long_stroke

def group_normalize(strokes):
    long_stroke = concat(strokes)
    x_min = min(long_stroke.x)
    x_max = max(long_stroke.x)
    y_min = min(long_stroke.y)
    y_max = max(long_stroke.y)
    x_range = float(x_max-x_min)
    y_range = float(y_max-y_min)
    normalized_strokes = []
    for stroke in strokes:
        x = ((np.array(stroke.x) - x_min)/x_range).tolist()
        y = ((np.array(stroke.y) - y_min)/y_range).tolist()
        normalized_strokes.append(Stroke(x,y))
    return normalized_strokes

def group_normalize_wrt_x(strokes):
    long_stroke = concat(strokes)
    x_min = min(long_stroke.x)
    x_max = max(long_stroke.x)
    y_min = min(long_stroke.y)
    x_range = float(x_max-x_min)
    normalized_strokes = []
    for stroke in strokes:
        x = ((np.array(stroke.x) - x_min)/x_range).tolist()
        y = ((np.array(stroke.y) - y_min)/x_range).tolist()
        normalized_strokes.append(Stroke(x,y))
    return normalized_strokes

def best_aligment(stroke1, stroke2, indice=None):
    """compare naive euclidian distance, smart euclidian distance 
       and smart euclidian distance after reverting one of the two strokes"""

 
    if indice and indice<len(stroke2.x):
        stroke2 = Stroke(stroke2.x[indice:],stroke2.y[indice:])

    if len(stroke1.x)>len(stroke2.x):
        stroke1 = Stroke(stroke1.x[:len(stroke2.x)], stroke1.y[:len(stroke2.y)])

    if len(stroke2.x)>len(stroke1.x):
        stroke2 = Stroke(stroke2.x[:len(stroke1.x)], stroke2.y[:len(stroke1.y)])

    (nx1,ny1,d1,d2,m1,m2) = align(stroke1,stroke2)
    (rx1,ry1,d3,d4,m3,m4) = align(stroke1.revert(),stroke2)

    if np.sum(d4)<np.sum(d2):
        nx1 = rx1
        ny1 = ry1
        d2 = d4
        m2 = m4

    if np.sum(d1)<np.sum(d2):
        nx1 = stroke1.x
        ny1 = stroke2.y
        d2 = d1
        m2 = m1

    return nx1,ny1,np.mean(d2),np.mean(m2)

def align(stroke1, stroke2):
    """aligne two strokes in order to compute 
       the euclidian distance between them in a smart way"""

    x1 = np.array(stroke1.x)
    x2 = np.array(stroke2.x)
    y1 = np.array(stroke1.y)
    y2 = np.array(stroke2.y)

    d = np.sqrt((x1-x2)**2+(y1-y2)**2)
    m = d-min(d)

    Ix1 = np.argmax(x1)
    Ix2 = np.argmax(x2)
    Iy1 = np.argmax(y1)
    Iy2 = np.argmax(y2)

    ix1 = np.argmin(x1)
    ix2 = np.argmin(x2)
    iy1 = np.argmin(y1)
    iy2 = np.argmin(y2)

    # rephasing :
    u = np.array([(Ix1-Ix2),(Iy1-Iy2),(ix1-ix2),(iy1-iy2)])
    indice_period = np.argmin(np.abs(u))
    period = u[indice_period]
    new_x1 = np.array(x1[period:].tolist() + x1[0:period].tolist())
    new_y1 = np.array(y1[period:].tolist() + y1[0:period].tolist())
    x1 = new_x1
    y1 = new_y1

    # resorting : if symetric part, revert it
    mx =  np.max((x1,x2),0)
    my =  np.max((y1,y2),0)
    sym_score = abs(x1-x2[::-1])+abs(y1-y2[::-1])
    if len(x1[sym_score<50])>20:
        x1[sym_score<40] = x1[sym_score<40][::-1]
        y1[sym_score<40] = y1[sym_score<40][::-1]

    new_d = np.sqrt((x1-x2)**2+(y1-y2)**2)
    new_m = new_d - min(new_d)

    return x1, y1, d, new_d, m, new_m

class Drawing:

    # strokes must be a collection of strokes never normalized

    def __init__(self, strokes):
        self.strokes = strokes
        self.strokes = group_normalize_wrt_x(strokes)

def identify(strokes, stroke):
    draw = concat(strokes)
    draw_length,_ = draw.euclidian_length()
    stroke_length,_ = stroke.euclidian_length()

    draw.uniformize()
    stroke.uniformize()

    numDesiredPoints = int(stroke.get_len()*float(draw_length)/float(stroke_length))
    draw.downsampleShape(numDesiredPoints)
    draw.len = len(draw.x)

    pose = 0
    _,_,best_score,best_match = best_aligment(stroke, draw, pose)
    for i in 1+np.array(range(draw.get_len()-stroke.get_len()+1)):
        _,_,score,match = best_aligment(stroke, draw, i)
        if match<best_match:
        #if score<best_score:
            best_score = score
            best_match = match
            pose = i

    #return pose, best_score, best_match

    print best_score

    plt.plot(draw.x,draw.y,'bo')
    plt.plot(draw.x[pose:pose+stroke.len],draw.y[pose:pose+stroke.len],'rs')
    plt.show()

def compare(strokes1, strokes2):
    score = 0
    for stroke in strokes1:
        _,_,match = identify(strokes2,stroke)
        score += match
    
    draw1 = concat(strokes1)
    draw2 = concat(strokes2)
    draw1_length,_ = draw1.euclidian_length()
    draw2_length,_ = draw2.euclidian_length()

    tot_length = draw1_length# + draw2_length

    return 100*score/tot_length

        

modelStrokes = []
demoStrokes = []
drawingCenters_x = []
drawingCenters_y = []
lastStroke = Stroke()
all_x = []
all_y = []

demo = False

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        with self.canvas:

            #self.canvas.clear()
            Color(1, 1, 0)
            d = 30.
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        global lastStroke
        touch.ud['line'].points += [touch.x, touch.y]
        lastStroke.append(touch.x, touch.y)

    def on_touch_up(self, touch):
        global demo
        global all_x
        global all_y
        global lastStroke
        global mainStroke
        global modelStrokes
        global demoStrokes
        global drawingCenters_x
        global drawingCenters_y
        touch.ud['line'].points

        if lastStroke.get_len() > 5:
            #lastStroke.uniformize()
            #lastStroke.downsampleShape(70)

            center = lastStroke.get_center() # before normalization
            all_x += lastStroke.get_x()
            all_y += lastStroke.get_y()

            #lastStroke.normalize_wrt_x()

            modelStrokes.append(lastStroke)

            """
            if demo:
                demoStrokes.append(lastStroke)
            else:
                modelStrokes.append(lastStroke)
            """

            drawingCenters_x.append(center[0])
            drawingCenters_y.append(center[1])
            
            lastStroke = Stroke()

        if touch.is_double_tap:
            #print demo

            """
            if demo:
                score = compare(demoStrokes,modelStrokes)
                print score
            """

            #print('Received demo')

            identify(modelStrokes[:-1], modelStrokes[-1])


            #x1, y1, score = best_aligment(modelStrokes[0],drawingStrokes[1])

            #print '...'
            #print sum(score)

            """
            if demo:
                demoStrokes = []
                modelStrokes = []
                self.canvas.clear()
            """

            modelStrokes = []
            self.canvas.clear()

            drawingCenters_x = []
            drawingCenters_y = []
            all_x = []
            all_y = []

            demo = not demo

            #mainStroke = []
            #self.canvas.clear()

            #showShape(learned_shape, 0)

class UserInputCapture(App):

    def build(self):
        self.painter = MyPaintWidget()
        return self.painter

    def on_start(self):
        with self.painter.canvas:
            print(self.painter.width)
            Color(1, 1, 0)
            d = 30.
 
            for i in range(len(wordToLearn)-1):
                x = (self.painter.width/len(wordToLearn))*(i+1)
                Line(points=(x, 0, x, self.painter.height))


def showShape(shape, shapeIndex):
    plt.figure(shapeIndex+1)
    plt.clf()
    ShapeModeler.normaliseAndShowShape(shape.path)

if __name__ == "__main__":
    args = parser.parse_args()
    wordToLearn = args.draw

   #import inspect
   #fileName = inspect.getsourcefile(ShapeModeler)
   #installDirectory = fileName.split('/lib')[0]
   ##datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/uji_pen_chars2'
   #init_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'
   #update_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'
   #demo_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/diego_set'
 
   #if not os.path.exists(init_datasetDirectory):
   #    raise RuntimeError("initial dataset directory not found !")
   #if not os.path.exists(update_datasetDirectory):
   #    os.makedir(update_datasetDirectory)

    #plt.ion()
    #for i in range(len(wordToLearn)):
        #showShape(shape, i)

    try:
        UserInputCapture().run()
        
    except KeyboardInterrupt:
            # ShapeModeler.save()
            logger.info("Bye bye")
