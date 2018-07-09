BASE_PATH=~/git_project/adaptive-system
export LD_LIBRARY_PATH=$BASE_PATH/build/
g++ -std=c++11 -I$BASE_PATH -I/home/cgx/include/ -I$BASE_PATH/tensorflow/ -I$BASE_PATH/eigen-eigen  synchronous_server_using_sarsa.cc sarsa.cc reward.cc ../proto/*.cc ../quantization/util/*.cc -o server_sarsa.bin -L$BASE_PATH/build/ -ltensorflow_framework -ltensorflow_cc -L/usr/local/lib -lgrpc++ -lgrpc  -lgrpc++_reflection -L/home/cgx/lib/ -lprotobuf -lpthread -ldl
chmod a+x server_sarsa.bin
