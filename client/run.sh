BASE_PATH=~/git_project/adaptive-system
export LD_LIBRARY_PATH=$BASE_PATH/build/
chmod a+x client.bin
./client.bin 10.61.1.120:50051 /home/cgx/git_project/adaptive-system/resources/cifar-10-batches-bin/data_batch.bin /home/cgx/git_project/adaptive-system/input/cifar10/preprocess.pb
