#!/bin/bash
service1="/lib/systemd/system/ApneaWS.service"
service2="/lib/systemd/system/ApneaTCP.service"
if test -e $service1
then
  echo "正在删除主程序服务……"
else
  echo "未找到主程序服务，请确认是否正常安装"
  exit
fi
rm -f $service1
if test -e $service1
then
  echo "删除失败，请确认脚本是否拥有权限（以 root 用户执行或采用 sudo 命令）"
else
  systemctl daemon-reload
  echo "删除成功"
fi
if test -e $service2
then
  echo "正在删除数据接收服务……"
else
  echo "未找到数据接收服务，请确认是否正常安装"
  exit
fi
rm -f $service2
if test -e $service2
then
  echo "删除失败，请确认脚本是否拥有权限（以 root 用户执行或采用 sudo 命令）"
else
  systemctl daemon-reload
  echo "删除成功"
fi