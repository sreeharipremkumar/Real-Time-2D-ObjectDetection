CC=g++
CFLAGS=-I.
LIBS = `pkg-config --cflags --libs opencv4`
LDLIBS = -ltiff -lpng -ljpeg -lz -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect

IDIR = ../include
_DEPS = OR_functions.h csv_util.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))


VID = vidDisplay.cpp OR_functions.o csv_util.o

all: vid_output 

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(LIBS)

vid_output: $(VID)
	$(CC) -o  $@ $^ $(LIBS)

clean:
	rm -f *.o *~

# CC = g++
# PROJECT = new_output
# SRC = test.cpp
# LIBS = `pkg-config --cflags --libs opencv4`
# $(PROJECT) : $(SRC)
# 	$(CC) $(SRC) -o $(PROJECT) $(LIBS)