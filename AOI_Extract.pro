TEMPLATE = app
CONFIG += console c++11 core
CONFIG -= app_bundle
#CONFIG -= qt

SOURCES += main.cpp

LIBS+= -L/home/chili/opencv-3.4.0/build-linux/install/lib -lopencv_core -lopencv_flann -lopencv_calib3d -lopencv_xfeatures2d -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_imgcodecs
INCLUDEPATH+= /home/chili/opencv-3.4.0/build-linux/install/include
