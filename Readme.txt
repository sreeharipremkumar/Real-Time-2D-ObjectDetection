Made by : Sreehari Premkumar ,MS Robotics Northeastern University

OS : Ubuntu 20.04 LTS
with VSCode and g++ compiler

To extract data from image set the code:
1) Open Terminal
2) Go to src folder within Project-3
3) Run "make" command

4) run ./vid_output
	If running at your end, need to change the video cam location from url to webcam
	
Once the stream is running we can : 
	1) Keep the object in front of the camera.
	2) Once it is detected, press 't' to train, enter 'y' to accept the object, then proceed to enter the Label.
	3) Once entered the new object will be recognized using that label.
	4) We can also adjust the threshold using ',' / '.' keys (increase/decrease) to adjust the detection in bad lighting
	5) Pressing 's' can save the image output for all the tasks into Results folder
	6) Pressing 'k' activates the KNN classification with k =3, we can see the effect of KNN, if we put a new object and train only 1 time with that.
	7) The normal mode will give the correct response, but KNN mode will mostly give false response since only 1 instance of that object is saved 
	and since K =3 some other object would be dominating the response.

Video Link :
https://drive.google.com/file/d/1hbfgaYsAeXN8xkHKjLgjbCVslJEGgGFV/view?usp=sharing

 
TIME TRAVEL:
                "I WONT BE USING MY TIME TRAVEL DAYS"
