rm facedetect
g++ -std=c++11 ../src/*.cpp -I /usr/local/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_videoio -lopencv_face -l sqlite3 -o facedetect
./facedetect
