export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
#format: interval, learning_rate, total_iter, number_wokers, init_level, tuple_path, rl_model_path, threshold
./server_sarsa.bin 5 0.2 2000 6 2 ~/git_project/adaptive-system/input/cifar10/tuple_gradient_descent.pb ~/git_project/adaptive-system/reinforcement_learning_model/sarsa_continous.pb 105000 
