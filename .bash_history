exit
cd mwp/
ls
cd ..
rm -rf mwp
mkdir mwp
cd mwp
ls
docker images
cd 
mkdir .local && mkdir .local/bin
cd .local/bin/
ls
chmod ugo+x ~/.local/bin/*
cd 
nano .bash_profile
cd mwp/
ls
docker build -t kaden/mwp .
nano Dockerfile 
docker build -t kaden/mwp .
lspci | grep -i nvidia
nano Dockerfile 
docker build -t kaden/mwp .
run-docker -it kaden/mwp bash
exit
cd mwp
ls
run-docker -it kaden/mwp bash
run-docker kaden/mwp test-for-gpus
test-for-gpus
lspci | grep -i nvidia
docker run --runtime=nvidia kaden/mwp 
docker run --runtime=nvidia kaden/mwp -it bash
docker run --runtime=nvidia -it kaden/mwp bash
exit
cd mwp/
ls
docker build -t kaden/mwp .
run-docker kaden/mwp
nvidia-docker -it kaden/mwp bash
nvidia-docker run -it kaden/mwp bash
nvidia-docker build -t kaden/mwp .
nvidia-docker run --runtime=nvidia -it kaden/mwp bash
ls -la /dev | grep nvidia
nvidia-docker run --runtime=nvidia --device /dev/nvidia0:/dev/nvidia0 -it kaden/mwp bash
docker run --runtime=nvidia --device /dev/nvidia0:/dev/nvidia0 -it kaden/mwp bash
exit
nvidia-docker
nvidia-docker --version
exit
cd mwp/
ls
nvidia-docker build -t kaden/mwp .
tmux attach
tmux new -s testing
docker cp kaden/mwp:~/reu/mwp/checkpoints/* ~/reu/
docker cp kaden/mwp:~/reu/mwp/checkpoints/train/* ~/reu/
docker cp kaden/mwp:reu/mwp/checkpoints/train/* ~/reu/
exit
tmux switch -t testing
tmux attach -t testing
exit
ls
nvidia-docker run -it -rm --runtime=nvidia kaden/mwp bash
nvidia-docker run -it --runtime=nvidia kaden/mwp bash
nvidia-docker build -t kaden/mwp .
nvidia-docker run -it --runtime=nvidia kaden/mwp bash
tmux detach
nvidia-docker run -it --runtime=nvidia kaden/mwp bash
exit
cd mwp/
ls
nvidia-docker build kaden/mwp .
nvidia-docker build -t kaden/mwp .
tmux attach -t testing
tmux detach
exit
rm -rf mwp/
ls
la
cd mwp/
ls
nvidia-docker build -t kaden/mwp .
docker images
nvidia-docker run -v ${PWD}:~/mwp -it --runtime=nvidia kaden/mwp bash
nvidia-docker run -v ${PWD}:/mwp -it --runtime=nvidia kaden/mwp bash
tmux attach -t testing
tmux list
tmux list-sessions
tmux new -t testing
tmux new -s training
exit
cd mwp/
nano transformer_mwp.py 
exit
cd mwp/
ls
nvidia-docker build -t kaden/mwp .
docker build -t kaden/mwp .
nvidia-docker build -t kaden/mwp .
cd 
cd ..
ls
cd home/kgriff12/
cd mwp
docker build -t kaden/mwp .
exit
cd mwp/
ls
docker build -t kaden/mwp .
cat /etc/group 
cd 
exit
cd mwp/
ls
docker build kaden/mwp .
docker build -t kaden/mwp .
nvidia-docker run -v ${PWD}:/mwp -it --runtime=nvidia kaden/mwp bash
nano transformer_tuning.py 
tmux attach -t tuning
tmux attach -s tuning
tmux attach -t testing
tmux new -s tuning
cd mwp/
ls
cd mwp
ls
ls logs
docker ps
docker exec -it c5df9c5abb61 /bin/bash
tmux ls
tmux a -t tuning
exit
watch -n 0.5 nvidia-smi
exit
cd mwp/
ls
docker build -t kaden/mwp .
tmux add -s training
tmux -s training
tmux session -t training
tmux new -s training
cd mwp/
nano transformer_keras.py 
exit
tmux a training
tmux a -t training
cd mwp/
ls
cat Dockerfile 
docker build -t kaden/mwp .
chmod a+x run.sh
chmod a+x tensorboard.sh
tmux a -t training
ls
tmux a -t training
ls
cd 
cd mwp/
ls
chmod a+x run 
chmod a+x tboard 
docker build -t kaden/mwp .
tmux a -t training
cd
ls
cd mwp/
ls
chmod a+x run
chmod a+x tboard
docker build -t kaden/mwp .
./run
cd
rm -rf mwp
mkdir mwp
cd mwp/
ls
chmod a+x run && chmod a+x tboard
docker build -t kaden/mwp .
cd
cd mw
cd mwp
docker build -t kaden/mwp .
chmod a+x run
chmod a+x tboard
./run
cd
rm -rf mwp
cd mwp/
ls
chmod a+x tboard
chmod a+x run
docker build -t kaden/mwp .
./run
cd
ls
cd mwp
ls
chmod a+x run
chmod a+x tboard
docker build -t kaden/mwp .
./run
tmux a -t training
cat transformer_keras.py 
docker build -t kaden/mwp .
tmux a -t training
docker build -t kaden/mwp .
tmux a -t training
docker build -t kaden/mwp .
tmux a -t training
docker build -t kaden/mwp .
tmux a -t training
