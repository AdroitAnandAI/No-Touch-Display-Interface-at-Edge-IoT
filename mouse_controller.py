'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui
import numpy as np

class MouseController:
    def __init__(self, precision, speed):
        precision_dict = {'high': 100, 'low': 1000, 'medium': 500}
        speed_dict = {'fast': 1, 'slow': 10, 'medium': 5}

        self.precision = precision_dict[precision]
        self.speed = speed_dict[speed]

        screenWidth, screenHeight = pyautogui.size()
        self.w = screenWidth
        self.h = screenHeight

        # self.big_num = np.inf
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        self.calibrated = False
        self.take = False

        pyautogui.FAILSAFE = False

    def moveRelative(self, direction):

        print("DIRECTION = " + str(direction))
        # Direction Index = ['left', 'right', 'up', 'down']
        if (direction == 0):
            pyautogui.move(-20, 0)
        elif (direction == 1):
            pyautogui.move(20, 0)
        elif (direction == 2):
            pyautogui.move(0, -20)
        elif (direction == 3):
            pyautogui.move(0, 20)

    def moveWithGaze(self, x, y):

        # To bound the x, y coordinates within screen.dimensions.
        x = max(min(x, self.x_max), self.x_min)
        y = max(min(y, self.y_max), self.y_min)

        # msg = 'x = ' + str(x) + " y = " + str(y) + \
        #                    " precision = " + str(self.precision)
        # print(msg)
        # pyautogui.alert(msg)

        # To compute the x, y coordinates in the screen.
        x_cord = self.w * (1 - (x - self.x_min) / (self.x_max - self.x_min))
        y_cord = self.h * (y - self.y_min) / (self.y_max - self.y_min)

        pyautogui.moveTo(x_cord, y_cord, duration=0.2)
        # pyautogui.moveRel(-x*self.precision, y*self.precision, duration=self.speed)

    def moveWithHead(self, x, y, headYawPitchBounds):

        # Fixing 40x40 box as yaw and pitch boundaries to
        # correspond to head turning left and right (yaw)
        # and also moving up and down (pitch)
        x_min = headYawPitchBounds[0]
        x_max = headYawPitchBounds[1]
        y_min = headYawPitchBounds[0]
        y_max = headYawPitchBounds[1]

        # To bound the x, y coordinates within screen.dimensions.
        x = max(min(x, x_max), x_min)
        y = max(min(y, y_max), y_min)

        # msg = 'x = ' + str(x) + " y = " + str(y) + \
        #                    " precision = " + str(self.precision)
        # print(msg)
        # pyautogui.alert(msg)

        # To compute the x, y coordinates in the screen.
        x_cord = self.w * (1 - (x - x_min) / (x_max - x_min))
        y_cord = self.h * (y - y_min) / (y_max - y_min)

        # print('x_cord: ' + str(x_cord))
        # print('y_cord: ' + str(y_cord))
        pyautogui.moveTo(x_cord, y_cord, duration=0.2)
        # pyautogui.moveRel(-x*self.precision, y*self.precision, duration=self.speed)

    def clickLeft(self):
        # pyautogui.click()
        return

    def clickRight(self):
        # pyautogui.click(button='right')
        return

    def drag(self):
        pyautogui.drag(0, 10, 1, button='left') 

    def scroll(self, value):
        # pyautogui.scroll(value) 
        return

    def write(self, txt):
        pyautogui.write(txt) 

    def captureCorners(self, x, y):

        if self.calibrated:
           return True

        if self.x_min is None:

            if self.take is False:

                pyautogui.alert("Look here...Top Right")
                # mc.move(self.w-1, 0)
                self.take = True
            else:
                self.x_min = x
                self.y_min = y
                self.take = False

        elif self.x_max is None:

            if self.take is False:
                pyautogui.alert("Look here...Bottom Left")
                # mc.move(0, self.h-1)
                self.take = True
            else:
                self.x_max = x
                self.y_max = y
                self.take = False
        else:
            self.calibrated = True

        return self.calibrated
