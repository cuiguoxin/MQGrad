BASE_PATH=~/git_project/adaptive-system
export LD_LIBRARY_PATH=$BASE_PATH/build/:/home/cgx/grpc/libs/opt/:/home/cgx/lib
g++ -std=c++11 -I$BASE_PATH/ -I/home/cgx/include/ -I$BASE_PATH/tensorflow/ -I$BASE_PATH/eigen-eigen  synchronous_client.cc ../proto/*.cc ../input/cifar10/input.cc ../quantization/util/*.cc -o client.bin -L$BASE_PATH/build/ -ltensorflow_framework -ltensorflow_cc -L/usr/local/lib -lgrpc++ -lgrpc  -lgrpc++_reflection -L/home/cgx/lib/ -lprotobuf -lpthread -ldl -lrt
chmod a+x client.bin
