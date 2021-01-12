#!/bin/sh  
TOM_HOME=$1 
ps -ef|grep $TOM_HOME|grep -v grep|grep -v kill  
if [ $? -eq 0 ];then  
kill -9 `ps -ef|grep $TOM_HOME|grep -v grep|grep -v kill|awk '{print $2}'`  
else  
echo $TOM_HOME' No Found Process'  
fi 