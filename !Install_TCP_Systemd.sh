#!/bin/bash
path="`pwd`/TCPServer.py"
service="/lib/systemd/system/ApneaTCP.service"

if test -e $service
then
  echo "文件已存在，请执行 !Uninstall.sh 脚本将文件删除后再进行操作"
  exit
fi

echo "请键入配置好环境的Python程序位置："
read python

echo "[Unit]" >> $service
echo "Description=呼吸暂停检测，数据接收程序" >> $service
echo "" >> $service
echo "[Service]" >> $service
echo "Type=simple" >> $service
echo "User=root" >> $service
echo "LimitNPROC=500" >> $service
echo "LimitNOFILE=1000000" >> $service
echo "ExecStart=$python $path" >> $service
echo "Restart=on-failure" >> $service
echo "StandardError=append:$error" >> $service
echo "" >> $service
echo "[Install]" >> $service
echo "WantedBy=multi-user.target" >> $service
if test -e $service
then
  systemctl daemon-reload
  systemctl enable ApneaTCP
  echo "已创建服务，请运行 systemctl start ApneaTCP 执行该服务"
  exit
else
  echo "服务创建失败，请确认脚本是否拥有权限（以 root 用户执行或采用 sudo 命令）"
  exit
fi