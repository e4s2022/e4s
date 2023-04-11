mkdir weights

wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/realesrnet_x4.pth
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth

mv *.pth ./weights